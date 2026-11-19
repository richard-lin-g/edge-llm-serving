#!/usr/bin/env python3
"""bench/d9_load.py — 并发压测器（纯 asyncio，零第三方依赖）

为什么不用 aiohttp：WSL 那套 Python 只保证 requests 在，压测器不该再让人装包。
这个客户端只做一件事——把 HTTP/1.1 的 SSE 流按行切开、打时间戳、算 QPS 和分位。

口径（必须和 D3 的 metrics.py 对齐，否则两张表不能放一起）：
  ttft_ms    发出请求 → 第一个非空 text 分片（含 TCP 连接 + 排队 + prefill）
  tpot_ms    (t_last - t_first) / (n_out - 1)，纯解码每 token
  e2e_ms     发出请求 → 流结束
  n_out      优先 usage.completion_tokens，其次 timings.predicted_n，最后数分片
  prompt_n   timings.prompt_n，只用来验证"每槽上下文够不够"，不用来算速度

五条压测纪律（少一条，数字就不可信）：
  1. ignore_eos=true + max_tokens 固定 → 每个请求输出等长，QPS/TPOT 才有分母
  2. cache_prompt=false + 每请求随机 nonce → 否则第 2 个请求命中前缀缓存，TTFT 假性偏低
  3. 每档先 warmup 再计时 → 第一批请求要付 kernel 加载/显存分配的钱
  4. 显存必须边跑边采 → 并发下 KV 是 -c × slots 量级，跑完再看已经回落
  5. 客户端自己也要计时 → 客户端先饱和，量到的是 Python 不是 GPU

用法：
  python3 bench/d9_load.py --port 8092 --np 4 --requests 100 \
      --out results/d9_requests.csv --meta results/d9_level_meta.csv
  python3 bench/d9_load.py --port 8092 --np 1 --requests 5 --warmup 1 --dry-run
"""
import argparse
import asyncio
import csv
import json
import os
import resource
import statistics
import subprocess
import sys
import threading
import time

PROMPT_PATH_DEFAULT = "bench/prompts_d9.jsonl"

REQ_FIELDS = ["ts", "tag", "np", "idx", "status", "http", "ttft_ms", "tpot_ms",
              "e2e_ms", "n_out", "prompt_n", "bytes_in", "err"]
META_FIELDS = ["ts", "tag", "np", "c_ctx", "ctx_slot", "requests", "ok", "fail",
               "wall_s", "agg_tps", "vram_base_mib", "vram_peak_mib",
               "gpu_util_avg", "gpu_util_max", "gpu_mem_util_max",
               "cpu_busy_avg", "client_cpu_s", "gpu_samples", "note"]


# ---------------------------------------------------------------- 采样
def nvidia_now():
    """一次读出利用率/显存/功耗。WSL 里能用；没有 GPU 的环境返回 None。

    逐字段容错：WSL 下 power.draw 常常是 [N/A]，不能因为一个字段把整帧采样丢了。
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,"
                            "utilization.memory,power.draw",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
        parts = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        g, m, mu = float(parts[0]), int(float(parts[1])), float(parts[2])
        p = None
        if len(parts) > 3 and parts[3] not in ("[N/A]", "N/A", ""):
            try:
                p = float(parts[3])
            except ValueError:
                p = None
        return g, m, mu, p
    except Exception:
        return None


def cpu_busy_ratio():
    """从 /proc/stat 读一次，返回累计 (busy, total) 以便做差分。"""
    try:
        with open("/proc/stat") as f:
            parts = [float(x) for x in f.readline().split()[1:]]
        total = sum(parts)
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0.0)  # idle + iowait
        return total - idle, total
    except Exception:
        return None


class Sampler(threading.Thread):
    """200 ms 一采：GPU 利用率/显存、整机 CPU。峰值和均值都要，别只报峰值。"""

    def __init__(self, interval_ms=200):
        super().__init__(daemon=True)
        self.interval = interval_ms / 1000.0
        self.samples = []           # (t, gpu_util, vram, mem_util, power)
        self.cpu_pairs = []         # (busy, total)
        self.baseline = nvidia_now()   # 进采样器先留一帧：跑得再短也有基线
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            v = nvidia_now()
            if v is not None:
                self.samples.append((time.perf_counter(),) + v)
            c = cpu_busy_ratio()
            if c is not None:
                self.cpu_pairs.append(c)
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=3)
        return self.summary()

    def summary(self):
        out = {"gpu_util_avg": None, "gpu_util_max": None, "gpu_mem_util_max": None,
               "vram_peak": None, "vram_base": None, "cpu_busy": None,
               "gpu_samples": len(self.samples)}
        if self.samples:
            utils = [s[1] for s in self.samples]
            out["gpu_util_avg"] = round(statistics.fmean(utils), 1)
            out["gpu_util_max"] = round(max(utils), 1)
            out["gpu_mem_util_max"] = round(max(s[3] for s in self.samples), 1)
            out["vram_peak"] = max(s[2] for s in self.samples)
            out["vram_base"] = self.samples[0][2]
        elif self.baseline:
            out["vram_base"] = self.baseline[1]
        if len(self.cpu_pairs) >= 2:
            db = self.cpu_pairs[-1][0] - self.cpu_pairs[0][0]
            dt = self.cpu_pairs[-1][1] - self.cpu_pairs[0][1]
            if dt > 0:
                out["cpu_busy"] = round(db / dt * 100, 1)
        return out


# ---------------------------------------------------------------- HTTP
async def read_headers(reader, timeout):
    status_line = await asyncio.wait_for(reader.readline(), timeout)
    if not status_line:
        raise RuntimeError("连接被关闭（没有状态行）")
    parts = status_line.split()
    if len(parts) < 2:
        raise RuntimeError(f"坏状态行: {status_line[:60]!r}")
    status = int(parts[1])
    headers = {}
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout)
        if line in (b"\r\n", b"\n", b""):
            break
        k, _, v = line.decode("latin-1").partition(":")
        headers[k.strip().lower()] = v.strip()
    return status, headers


async def body_blocks(reader, headers, timeout):
    """按 chunked 或 content-length 吐字节块；只管传输层，不解析 SSE。"""
    if "chunked" in headers.get("transfer-encoding", "").lower():
        while True:
            size_line = await asyncio.wait_for(reader.readline(), timeout)
            if not size_line:
                return
            size = int(size_line.split(b";")[0].strip() or b"0", 16)
            if size == 0:
                while True:                      # 吃掉 trailer
                    line = await asyncio.wait_for(reader.readline(), timeout)
                    if line in (b"\r\n", b"\n", b""):
                        return
            left = size
            while left > 0:
                block = await asyncio.wait_for(reader.readexactly(min(left, 65536)), timeout)
                left -= len(block)
                yield block
            await asyncio.wait_for(reader.readexactly(2), timeout)
    else:
        left = int(headers.get("content-length") or 0)
        if left:
            while left > 0:
                block = await asyncio.wait_for(reader.readexactly(min(left, 65536)), timeout)
                left -= len(block)
                yield block
        else:
            while True:
                block = await asyncio.wait_for(reader.read(65536), timeout)
                if not block:
                    return
                yield block


class SSE:
    """最小 SSE 解析：data: 行 → JSON；只关心 text、usage、timings。"""

    def __init__(self):
        self.buf = b""
        self.first = self.last = None
        self.n_text = 0
        self.usage = None
        self.timings = None
        self.n_bytes = 0
        self.error = None

    def feed(self, data):
        self.n_bytes += len(data)
        self.buf += data
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            self._line(line.strip())

    def close(self):
        if self.buf:
            self._line(self.buf.strip())
            self.buf = b""

    def _line(self, line):
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if payload in (b"[DONE]", b""):
            return
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return
        if isinstance(obj.get("error"), dict):
            self.error = obj["error"].get("message") or str(obj["error"])
            return
        if isinstance(obj.get("usage"), dict):
            self.usage = obj["usage"]
        if isinstance(obj.get("timings"), dict):
            self.timings = obj["timings"]
        txt = obj.get("content")                      # /completion 原生
        if txt is None:                               # /v1/completions
            for ch in obj.get("choices") or []:
                t = ch.get("text")
                if t:
                    txt = (txt or "") + t
        if txt:
            now = time.perf_counter()
            if self.first is None:
                self.first = now
            self.last = now
            self.n_text += 1


async def one_request(host, port, body_bytes, timeout):
    t0 = time.perf_counter()
    sse = SSE()
    http = 0
    reader = writer = None
    try:
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(body_bytes)
        await writer.drain()
        http, headers = await read_headers(reader, timeout)
        if http != 200:
            raise RuntimeError(f"HTTP {http}")
        async for block in body_blocks(reader, headers, timeout):
            sse.feed(block)
        sse.close()
        if sse.error:
            raise RuntimeError(str(sse.error))
        t_end = time.perf_counter()
        n_out, src = 0, "events"
        if sse.usage and sse.usage.get("completion_tokens"):
            n_out, src = int(sse.usage["completion_tokens"]), "usage"
        elif sse.timings and sse.timings.get("predicted_n"):
            n_out, src = int(sse.timings["predicted_n"]), "timings"
        else:
            n_out = sse.n_text
        if sse.first is None:
            raise RuntimeError("一个 token 都没收到（prompt 被截断？ctx 太小？）")
        ttft = (sse.first - t0) * 1000
        tpot = None
        if n_out > 1 and sse.last:
            tpot = (sse.last - sse.first) * 1000 / (n_out - 1)
        prompt_n = int(sse.timings["prompt_n"]) if sse.timings and sse.timings.get("prompt_n") else None
        return {"status": "ok", "http": http, "ttft_ms": round(ttft, 1),
                "tpot_ms": round(tpot, 3) if tpot else None,
                "e2e_ms": round((t_end - t0) * 1000, 1), "n_out": n_out,
                "prompt_n": prompt_n, "bytes_in": sse.n_bytes, "err": "", "src": src}
    except Exception as e:
        return {"status": "fail", "http": http, "ttft_ms": None, "tpot_ms": None,
                "e2e_ms": round((time.perf_counter() - t0) * 1000, 1), "n_out": 0,
                "prompt_n": None, "bytes_in": sse.n_bytes, "src": "events",
                "err": f"{type(e).__name__}: {e}"[:120]}
    finally:
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


# ---------------------------------------------------------------- 调度
def build_body(model, prompt, out_tokens):
    payload = {
        "model": model, "prompt": prompt, "max_tokens": out_tokens,
        "temperature": 0, "stream": True, "ignore_eos": True,
        "cache_prompt": False, "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload, ensure_ascii=False).encode()
    return (f"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            "Content-Type: application/json\r\nAccept: text/event-stream\r\n"
            "Connection: close\r\n"
            f"Content-Length: {len(body)}\r\n\r\n").encode() + body


async def run_round(host, port, np, idxs, prompt_text, model, out_tokens,
                    timeout, tag, nonce_seed, collect):
    queue = asyncio.Queue()
    for i in idxs:
        queue.put_nowait(i)
    rows = []

    async def worker(wid):
        while True:
            try:
                idx = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            nonce = f"{nonce_seed}/{tag}/{np}/{idx}/{os.urandom(3).hex()}"
            prompt = f"[{nonce}]\n{prompt_text}\n"      # nonce 打头 → 前缀缓存失效
            body = build_body(model, prompt, out_tokens)
            row = await one_request(host, port, body, timeout)
            row.update({"idx": idx, "np": np, "tag": tag,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
            if collect:
                rows.append(row)
            if not collect and row["status"] != "ok":
                print(f"  warmup 失败: {row['err']}", file=sys.stderr)

    await asyncio.gather(*(worker(i) for i in range(np)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--np", type=int, required=True, help="并发数 = 服务端 -np")
    ap.add_argument("--requests", type=int, default=100, help="本档总请求数")
    ap.add_argument("--warmup", type=int, default=4, help="先跑几笔热身（不计入统计）")
    ap.add_argument("--out-tokens", type=int, default=64, help="max_tokens，固定输出长度")
    ap.add_argument("--timeout", type=float, default=120.0, help="单请求读超时（秒）")
    ap.add_argument("--tag", default="d9")
    ap.add_argument("--model", default="local")
    ap.add_argument("--prompt-file", default=PROMPT_PATH_DEFAULT)
    ap.add_argument("--c-ctx", type=int, default=None, help="服务端 -c（写进 meta，便于对账）")
    ap.add_argument("--ctx-slot", type=int, default=None, help="期望的每槽上下文")
    ap.add_argument("--out", default="results/d9_requests.csv")
    ap.add_argument("--meta", default="results/d9_level_meta.csv")
    ap.add_argument("--dry-run", action="store_true", help="只打 3 个请求验证链路")
    args = ap.parse_args()

    if args.dry_run:
        args.requests, args.warmup = 3, 0

    with open(args.prompt_file, encoding="utf-8") as f:
        first = json.loads(next(line for line in f if line.strip()))
    prompt_text = first["text"]
    if int(first.get("repeat", 1)) > 1:
        prompt_text = "\n".join(f"[段{i + 1}] {prompt_text}" for i in range(int(first["repeat"])))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.meta) or ".", exist_ok=True)

    print(f"[np={args.np}] 热身 {args.warmup} 笔 → 正式 {args.requests} 笔，"
          f"输出固定 {args.out_tokens} token，超时 {args.timeout}s")
    if args.warmup:
        asyncio.run(run_round(args.host, args.port, args.np, list(range(args.warmup)),
                              prompt_text, args.model, args.out_tokens, args.timeout,
                              args.tag, "warm", collect=False))

    sampler = Sampler()
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    sampler.start()
    time.sleep(0.15)          # 让采样线程先落一帧，再开始计时
    rows = asyncio.run(run_round(args.host, args.port, args.np,
                                 list(range(args.requests)), prompt_text, args.model,
                                 args.out_tokens, args.timeout, args.tag, "run",
                                 collect=True))
    wall = time.perf_counter() - t0
    s = sampler.stop()
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    client_cpu = (cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime)

    rows.sort(key=lambda r: r["idx"])
    ok = [r for r in rows if r["status"] == "ok"]
    fail = len(rows) - len(ok)
    agg_tps = sum(r["n_out"] for r in ok) / wall if wall > 0 else 0
    pns = [r["prompt_n"] for r in ok if r["prompt_n"]]
    note = ""
    if pns and args.ctx_slot and statistics.median(pns) + args.out_tokens > args.ctx_slot:
        note = "PROMPT_TRUNCATED"       # prompt+输出超过每槽上下文，数字作废
    if s["cpu_busy"] and s["cpu_busy"] > 85:
        note = (note + ";CLIENT_CPU_HIGH").strip(";")

    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REQ_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)

    meta = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": args.tag, "np": args.np,
        "c_ctx": args.c_ctx, "ctx_slot": args.ctx_slot, "requests": len(rows),
        "ok": len(ok), "fail": fail, "wall_s": round(wall, 2),
        "agg_tps": round(agg_tps, 2),
        "vram_base_mib": s["vram_base"], "vram_peak_mib": s["vram_peak"],
        "gpu_util_avg": s["gpu_util_avg"], "gpu_util_max": s["gpu_util_max"],
        "gpu_mem_util_max": s["gpu_mem_util_max"],
        "cpu_busy_avg": s["cpu_busy"], "client_cpu_s": round(client_cpu, 2),
        "gpu_samples": s["gpu_samples"], "note": note,
    }
    new = not os.path.exists(args.meta)
    with open(args.meta, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(meta)

    p50 = statistics.median([r["e2e_ms"] for r in ok]) if ok else 0
    print(f"[np={args.np}] 墙钟 {wall:.1f}s  QPS {len(ok) / wall:.2f}  "
          f"P50 {p50:.0f} ms  聚合 {agg_tps:.1f} tok/s  失败 {fail}")
    if s["vram_peak"]:
        print(f"  显存 {s['vram_base']} → 峰值 {s['vram_peak']} MiB  "
              f"GPU util 均值 {s['gpu_util_avg']}%  max {s['gpu_util_max']}%")
    else:
        print("  没读到 nvidia-smi：显存/GPU 利用率这一列为空（WSL 里要在 PATH 能调 nvidia-smi）")
    if pns:
        print(f"  prompt_n 中位数 {int(statistics.median(pns))} / 每槽 ctx {args.ctx_slot or '?'}")
    if note:
        print(f"  ! 口径异常: {note}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
