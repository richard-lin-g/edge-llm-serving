#!/usr/bin/env python3
"""bench/metrics.py — 一把尺子量延迟与吞吐，llama-server / vLLM 通用。

口径（跟别人比数之前，先比口径）：
  ttft_ms     发出请求 → 第一个 content 分片（含 HTTP 往返 + prefill）
  tpot_ms     (t_last - t_first) / (n_out - 1)，纯解码每 token
  decode_tps  1000 / tpot_ms
  e2e_tps     n_out / (t_end - t0)
  srv_pp_tps  服务端自报 prefill：timings.prompt_per_second
  srv_tg_tps  服务端自报解码：timings.predicted_per_second  ← 只有它俩能和 llama-bench 比
  vram_peak   250 ms 轮询 nvidia-smi 的峰值（含桌面基线）

用法：
  python bench/metrics.py --tag a-3b-ngl36 --base-url http://127.0.0.1:8081 \\
      --prompts bench/prompts.jsonl --repeat 3 --out results/d3_metrics.csv
  python bench/metrics.py --check results/d3_metrics.csv
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import requests

# 三个锚点：D2 的实测值。改这里就能重判。
ANCHORS = {
    "a-3b-ngl36": {"tg": 39.33, "pp": 152.14},
    "b-7b-ngl22": {"tg": 16.04},
}
TOL = 10.0  # 容忍误差百分比


def median_int(vals):
    return int(statistics.median(vals)) if vals else 0

FIELDS = ["ts", "tag", "tier", "prompt_id", "rep", "n_out", "pp_n", "ttft_ms",
          "tpot_ms", "decode_tps", "e2e_tps", "srv_pp_tps", "srv_tg_tps",
          "vram_peak_mib", "vram_base_mib", "tok_src"]


def vram_now():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=3)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class VramSampler(threading.Thread):
    """后台每 250 ms 取一次显存，记住峰值。峰值含桌面基线，报数时必须交代。"""

    def __init__(self, interval_ms=250):
        super().__init__(daemon=True)
        self.interval = interval_ms / 1000.0
        self.peak = 0
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            v = vram_now()
            if v is not None:
                self.peak = max(self.peak, v)
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=3)
        return self.peak or None


def build_prompt(entry, nonce):
    """nonce 打头：每次请求前缀都不同，彻底废掉 prompt cache。"""
    text = entry["text"]
    rep = int(entry.get("repeat", 1))
    if rep > 1:
        text = "\n".join(f"[段{i + 1}] {text}" for i in range(rep))
    return f"[{nonce}]\n{text}\n"


def parse_stream(resp):
    t_first = t_last = None
    n_events = 0
    timings = usage = None
    for raw in resp.iter_lines():
        if not raw or not raw.startswith(b"data: "):
            continue
        payload = raw[6:].strip()
        if payload == b"[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj.get("timings"), dict):
            timings = obj["timings"]
        if isinstance(obj.get("usage"), dict):
            usage = obj["usage"]
        for ch in obj.get("choices") or []:
            if ch.get("text"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                n_events += 1
    return t_first, t_last, n_events, timings, usage


def one_run(base_url, model, prompt, n_predict):
    sampler = VramSampler()
    sampler.start()
    time.sleep(0.3)  # 先让采样器拿到一帧基线
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": n_predict,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,                 # 输出恒为 n_predict 个 token
        "cache_prompt": False,              # llama-server 专有；vLLM 会忽略
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{base_url}/v1/completions", json=body,
                         stream=True, timeout=(5, 600))
    resp.raise_for_status()
    t_first, t_last, n_events, timings, usage = parse_stream(resp)
    resp.close()
    t_end = time.perf_counter()
    peak = sampler.stop()

    if usage and usage.get("completion_tokens"):
        n_out, src = int(usage["completion_tokens"]), "usage"
    elif timings and timings.get("predicted_n"):
        n_out, src = int(timings["predicted_n"]), "timings"
    else:
        n_out, src = n_events, "events"

    ttft = (t_first - t0) * 1000 if t_first else None
    tpot = None
    if t_first and t_last and n_out and n_out > 1:
        tpot = (t_last - t_first) * 1000 / (n_out - 1)

    def r(v, n=2):
        return round(v, n) if v is not None else None

    return {
        "n_out": n_out,
        "pp_n": int(timings["prompt_n"]) if timings and timings.get("prompt_n") else None,
        "ttft_ms": r(ttft, 1),
        "tpot_ms": r(tpot, 3),
        "decode_tps": r(1000 / tpot) if tpot else None,
        "e2e_tps": r(n_out / (t_end - t0)) if n_out else None,
        "srv_pp_tps": r(timings.get("prompt_per_second")) if timings else None,
        "srv_tg_tps": r(timings.get("predicted_per_second")) if timings else None,
        "vram_peak_mib": peak,
        "tok_src": src,
    }


def run(args):
    with open(args.prompts, encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    base = vram_now()
    print(f"[base] 空闲显存 {base} MiB —— 峰值含这段基线，报数时要交代")
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for entry in prompts:
            for rep in range(1, args.repeat + 1):
                nonce = f"{args.tag}/{entry['id']}/r{rep}/{os.urandom(3).hex()}"
                row = one_run(args.base_url, args.model,
                              build_prompt(entry, nonce), args.n_predict)
                row.update({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "tag": args.tag,
                            "tier": entry.get("tier", "?"),
                            "prompt_id": entry["id"],
                            "rep": rep,
                            "vram_base_mib": base})
                w.writerow(row)
                fh.flush()
                print(f"{entry['id']} r{rep}  pp_n {row['pp_n']}  "
                      f"ttft {row['ttft_ms']} ms  decode {row['decode_tps']} t/s  "
                      f"srv_tg {row['srv_tg_tps']}  peak {row['vram_peak_mib']} MiB")


def check(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("空表：先跑 --tag 那一步")

    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["tag"], []).append(r)

    print(f"{'tag':13s} {'tier':4s} {'n':>3s} {'pp_n':>5s} {'decode_tps':>11s} {'CV%':>6s} "
          f"{'ttft_ms':>8s} {'CV%':>6s} {'peak':>6s}")
    all_ok = True
    for tag, rs in by_tag.items():
        for tier in sorted({r["tier"] for r in rs}):
            trs = [r for r in rs if r["tier"] == tier and r["decode_tps"]]
            if not trs:
                continue
            dec = [float(r["decode_tps"]) for r in trs]
            ttf = [float(r["ttft_ms"]) for r in trs if r["ttft_ms"]]
            cv_d = statistics.pstdev(dec) / statistics.mean(dec) * 100 if len(dec) > 1 else 0.0
            cv_t = statistics.pstdev(ttf) / statistics.mean(ttf) * 100 if len(ttf) > 1 else 0.0
            peaks = [int(r["vram_peak_mib"]) for r in trs if r["vram_peak_mib"]]
            ppn = [int(r["pp_n"]) for r in trs if r["pp_n"]]
            if cv_d > TOL:
                all_ok = False
            print(f"{tag:13s} {tier:4s} {len(trs):3d} {statistics.median(ppn) if ppn else 0:5.0f} "
                  f"{statistics.median(dec):11.2f} "
                  f"{cv_d:6.1f} {statistics.median(ttf) if ttf else 0:8.1f} "
                  f"{cv_t:6.1f} {max(peaks) if peaks else 0:6d}")

    print("\n锚点复现（只取 L 档：锚点来自 pp512 上下文，短 prompt 的 KV 小、tg 会偏高）：")
    for tag, want in ANCHORS.items():
        all_rs = [r for r in by_tag.get(tag, []) if r["srv_tg_tps"]]
        rs = [r for r in all_rs if r["tier"] == "L"] or all_rs
        if not rs:
            print(f"  {tag}: 无数据 → 未判定（是不是降了 ngl？记口径偏离）")
            all_ok = False
            continue
        ok = True
        tg = statistics.median([float(r["srv_tg_tps"]) for r in rs])
        ppn = [int(r["pp_n"]) for r in rs if r["pp_n"]]
        line = f"  {tag}: n={len(rs)} pp_n={median_int(ppn)} tg {tg:.2f}"
        if want.get("tg"):
            err = abs(tg - want["tg"]) / want["tg"] * 100
            ok = ok and err <= TOL
            line += f" vs {want['tg']} → {err:.1f}%"
        pp = [float(r["srv_pp_tps"]) for r in rs if r["srv_pp_tps"]]
        if want.get("pp") and pp:
            med = statistics.median(pp)
            err = abs(med - want["pp"]) / want["pp"] * 100
            ok = ok and err <= TOL
            line += f" | pp {med:.2f} vs {want['pp']} → {err:.1f}%"
        print(line + ("  PASS" if ok else "  FAIL"))
        all_ok = all_ok and ok

    sys.exit(0 if all_ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="local")
    ap.add_argument("--tag", default="a-3b-ngl36")
    ap.add_argument("--prompts", default="bench/prompts.jsonl")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default="results/d3_metrics.csv")
    ap.add_argument("--check", metavar="CSV", help="只做汇总判定，不发请求")
    a = ap.parse_args()
    if a.check:
        check(a.check)
    else:
        run(a)


if __name__ == "__main__":
    main()
