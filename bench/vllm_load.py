#!/usr/bin/env python3
"""bench/vllm_load.py — 小规模对照压测（vLLM 端，纯 asyncio，零第三方依赖）

为什么另写一个而不是复用 d9_load.py：d9 那个发的是 llama.cpp 的
`/v1/completions`，带 `cache_prompt`、`n_predict` 这类 llama 专有字段；
vLLM 的 OpenAI server 对未知字段更严格，而且它认的是 `max_tokens` / `ignore_eos`。
口径照旧对齐（ttft / tpot / e2e / n_out / prompt_n），但**表不能和 llama.cpp 并排放**：
精度档不同（AWQ W4A16 vs GGUF Q4_K_M），并排就是错的口径。

用法：
  python3 bench/vllm_load.py --port 8100 --model qwen25-3b-awq --health
  python3 bench/vllm_load.py --port 8100 --model qwen25-3b-awq \
      --np-list 1,2 --requests 20 --max-tokens 64 \
      --out results/d10_requests.csv
"""
import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
import urllib.request

FIELDS = ["ts", "tag", "np", "idx", "status", "http", "ttft_ms", "tpot_ms",
          "e2e_ms", "n_out", "prompt_n", "bytes_in", "err"]

PROMPT = ("You are an industrial equipment assistant. A CNC machine reports "
          "alarm code E-1042 at spindle startup. Explain in one short paragraph "
          "what to check first and why.")


async def read_headers(reader, timeout):
    line = await asyncio.wait_for(reader.readline(), timeout)
    if not line:
        raise RuntimeError("连接被关闭（无状态行）")
    status = int(line.split()[1])
    headers = {}
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout)
        if line in (b"\r\n", b"\n", b""):
            break
        k, _, v = line.decode("latin-1").partition(":")
        headers[k.strip().lower()] = v.strip()
    return status, headers


async def body_blocks(reader, headers, timeout):
    if "chunked" in headers.get("transfer-encoding", "").lower():
        while True:
            size_line = await asyncio.wait_for(reader.readline(), timeout)
            if not size_line:
                return
            size = int(size_line.split(b";")[0].strip() or b"0", 16)
            if size == 0:
                while True:
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


def build_body(model, prompt, max_tokens, ignore_eos):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True      # vLLM 认这个扩展字段，输出等长才有分母
    body = json.dumps(payload, ensure_ascii=False).encode()
    head = (f"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            "Content-Type: application/json\r\nAccept: text/event-stream\r\n"
            "Connection: close\r\n"
            f"Content-Length: {len(body)}\r\n\r\n").encode()
    return head + body


async def one_request(host, port, body, timeout):
    t0 = time.perf_counter()
    first = last = None
    n_text = 0
    usage = None
    err = ""
    http = 0
    nbytes = 0
    cont = b""
    writer = None
    try:
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(body)
        await writer.drain()
        http, headers = await read_headers(reader, timeout)
        if http != 200:
            raise RuntimeError(f"HTTP {http}")
        async for block in body_blocks(reader, headers, timeout):
            nbytes += len(block)
            cont += block
            while b"\n" in cont:
                line, cont = cont.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload in (b"[DONE]", b""):
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj.get("usage"), dict):
                    usage = obj["usage"]
                if isinstance(obj.get("error"), dict):
                    err = obj["error"].get("message") or str(obj["error"])
                for ch in obj.get("choices") or []:
                    if ch.get("text"):
                        now = time.perf_counter()
                        first = first or now
                        last = now
                        n_text += 1
        t_end = time.perf_counter()
        if err:
            raise RuntimeError(err)
        if first is None:
            raise RuntimeError("一个 token 都没收到")
        n_out = int(usage["completion_tokens"]) if usage and usage.get("completion_tokens") else n_text
        ttft = (first - t0) * 1000
        tpot = (last - first) * 1000 / (n_out - 1) if n_out > 1 and last else None
        return {"status": "ok", "http": http, "ttft_ms": round(ttft, 1),
                "tpot_ms": round(tpot, 3) if tpot else None,
                "e2e_ms": round((t_end - t0) * 1000, 1), "n_out": n_out,
                "prompt_n": None, "bytes_in": nbytes, "err": ""}
    except Exception as e:
        return {"status": "fail", "http": http, "ttft_ms": None, "tpot_ms": None,
                "e2e_ms": round((time.perf_counter() - t0) * 1000, 1), "n_out": 0,
                "prompt_n": None, "bytes_in": nbytes,
                "err": f"{type(e).__name__}: {e}"[:120]}
    finally:
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


async def run_level(host, port, np_, n_req, model, max_tokens, timeout, tag, warmup):
    q = asyncio.Queue()
    for i in range(n_req):
        q.put_nowait(i)
    rows = []

    async def worker():
        while True:
            try:
                idx = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            prompt = f"[{os.urandom(4).hex()}]\n{PROMPT}"   # 随机前缀 → 前缀缓存失效
            body = build_body(model, prompt, max_tokens, True)
            row = await one_request(host, port, body, timeout)
            row.update({"idx": idx, "np": np_,
                        "tag": tag, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
            rows.append(row)

    t0 = time.perf_counter()
    for _ in range(warmup):
        await one_request(host, port, build_body(model, PROMPT, 8, True), timeout)
    await asyncio.gather(*(worker() for _ in range(np_)))
    wall = time.perf_counter() - t0

    ok = [r for r in rows if r["status"] == "ok"]
    fails = [r for r in rows if r["status"] != "ok"]
    e2e = sorted(r["e2e_ms"] for r in ok)
    ttft = sorted(r["ttft_ms"] for r in ok)

    def pct(xs, p):
        if not xs:
            return None
        k = max(1, int(round(p / 100 * len(xs))))     # 最近秩，不插值（与 D9 一致）
        return xs[min(k, len(xs)) - 1]

    summary = {
        "np": np_, "requests": len(rows), "ok": len(ok), "fail": len(fails),
        "wall_s": round(wall, 1),
        "qps": round(len(ok) / wall, 3) if wall else 0,
        "p50": pct(e2e, 50), "p95": pct(e2e, 95),
        "ttft_p50": pct(ttft, 50), "ttft_p95": pct(ttft, 95),
        "agg_tps": round(sum(r["n_out"] for r in ok) / wall, 1) if wall else 0,
        "n_out_set": sorted({r["n_out"] for r in ok}),
        "err_sample": fails[0]["err"] if fails else "",
    }
    print(f"  np={np_}: ok {len(ok)}/{len(rows)}  QPS {summary['qps']}  "
          f"P50 {summary['p50']} ms  P95 {summary['p95']} ms  "
          f"聚合 {summary['agg_tps']} tok/s  n_out={summary['n_out_set']}")
    if len(summary["n_out_set"]) > 1:
        print("  ! n_out 不恒定：ignore_eos 可能被拒，QPS/TPOT 分母不一致，这一档只能参考")
    if fails:
        print(f"  ! 失败样例: {summary['err_sample']}")
    return rows, summary


def health(host, port):
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=5) as r:
            obj = json.load(r)
        print("服务在线：", [m.get("id") for m in obj.get("data", [])])
        return True
    except Exception as e:
        print(f"服务不在线：{type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--model", default="qwen25-3b-awq", help="/v1/models 里的 id 或你起的 --served-model-name")
    ap.add_argument("--np-list", default="1,2")
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--tag", default="d10")
    ap.add_argument("--out", default="results/d10_requests.csv")
    ap.add_argument("--health", action="store_true", help="只探活，不压")
    args = ap.parse_args()

    if args.health:
        sys.exit(0 if health(args.host, args.port) else 3)

    nps = [int(x) for x in args.np_list.split(",") if x.strip()]
    out_rows = []
    for np_ in nps:
        rows, _ = asyncio.run(run_level(args.host, args.port, np_, args.requests,
                                        args.model, args.max_tokens, args.timeout,
                                        args.tag, args.warmup))
        out_rows += rows

    if out_rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        write_header = not os.path.exists(args.out)
        with open(args.out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            for r in out_rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})
        print(f"\n已写入 {args.out}（{len(out_rows)} 行）")


if __name__ == "__main__":
    main()
