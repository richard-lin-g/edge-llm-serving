#!/usr/bin/env python3
"""D10 vLLM 小压测：SSE 流式，记录 TTFT / 端到端延迟 / 输出 tok/s / 显存与 util 峰值。
用法：python bench/d10_bench.py --tag L1 --conc 1,2,4 --reqs 6 --max-tokens 64
"""
import argparse, json, os, queue, subprocess, threading, time, urllib.request, csv, statistics

PROMPT = ("你是一名边缘推理工程师。请用中文解释下面三个问题，条理清晰，每条不超过两句话："
          "1) PagedAttention 与传统 KV cache 预分配的区别；"
          "2) continuous batching 相比静态 batching 提升了什么；"
          "3) 在 4 GB 显存上跑 3B 模型时最该先调哪个参数。")

def pct(xs, q):
    if not xs: return float("nan")
    xs = sorted(xs); k = max(0, min(len(xs)-1, int(round((len(xs)-1)*q))))
    return xs[k]

class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.stop = False
        self.mem = []; self.util = []
    def run(self):
        while not self.stop:
            try:
                out = subprocess.run(["nvidia-smi","--query-gpu=memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
                m, u = [x.strip() for x in out.split(",")[:2]]
                self.mem.append(int(m)); self.util.append(int(u))
            except Exception:
                pass
            time.sleep(0.8)

def one(port, model, max_tokens):
    body = json.dumps({"model": model, "stream": True, "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": PROMPT}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; n = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            try: o = json.loads(d)
            except Exception: continue
            if o["choices"][0].get("delta", {}).get("content"):
                if ttft is None: ttft = time.perf_counter() - t0
                n += 1
    return ttft, time.perf_counter() - t0, n

def run(port, model, conc, reqs, max_tokens, tag):
    q = queue.Queue(); [q.put(i) for i in range(reqs)]
    res = []; lock = threading.Lock()
    def worker():
        while True:
            try: q.get_nowait()
            except queue.Empty: return
            try:
                r = one(port, model, max_tokens)
                with lock: res.append(r)
            except Exception as e:
                with lock: res.append((None, None, f"ERR:{e}"))
    ss = Sampler(); ss.start()
    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(conc)]
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.perf_counter() - t0
    ss.stop = True
    ok = [r for r in res if isinstance(r[2], int)]
    err = [r for r in res if not isinstance(r[2], int)]
    ttfts = [r[0] for r in ok]; lats = [r[1] for r in ok]; toks = [r[2] for r in ok]
    out_tok = sum(toks)
    decode = [(r[2] / (r[1] - r[0])) if r[1] > r[0] else 0.0 for r in ok]
    row = dict(tag=tag, conc=conc, reqs=reqs, ok=len(ok), err=len(err),
               qps=round(len(ok)/wall, 3),
               ttft_p50=round(pct(ttfts,0.50)*1000, 1), ttft_p95=round(pct(ttfts,0.95)*1000, 1),
               lat_p50=round(pct(lats,0.50)*1000, 1), lat_p95=round(pct(lats,0.95)*1000, 1),
               out_tok_s=round(out_tok/wall, 1), per_stream_tok_s=round(statistics.mean(decode), 1),
               mem_peak_mib=max(ss.mem) if ss.mem else None, util_peak=max(ss.util) if ss.util else None,
               max_tokens=max_tokens, note="chunk≈1 token（SSE 无 usage，输出 token 数按 chunk 计）")
    print(json.dumps(row, ensure_ascii=False))
    return row

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen25-3b-awq")
    ap.add_argument("--conc", default="1,2,4")
    ap.add_argument("--reqs", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--tag", default="L1")
    ap.add_argument("--csv", default="results/d10_vllm_bench.csv")
    a = ap.parse_args()
    rows = [run(a.port, a.model, int(c), a.reqs, a.max_tokens, a.tag) for c in a.conc.split(",")]
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new: w.writeheader()
        [w.writerow(r) for r in rows]