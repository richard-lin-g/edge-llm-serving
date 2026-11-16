#!/usr/bin/env python3
"""bench/d7_speed_to_bench.py — 把 metrics.py 的三次重复压成两行表（pp / tg 中位数）

用法：
  python3 bench/d7_speed_to_bench.py --csv results/d7_speed_q4_0.csv \
      --quant q4_0 --ngl 36 --file qwen2.5-3b-instruct-q4_0.gguf \
      --out results/d7_quant_bench.csv
"""
import argparse
import csv
import os
import statistics

FIELDS = ["quant", "file", "size_gib", "ngl", "test", "test_tps", "stddev", "commit", "host"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--ngl", required=True)
    ap.add_argument("--file", default="")
    ap.add_argument("--size", default="")
    ap.add_argument("--out", default="results/d7_quant_bench.csv")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    pp = [float(r["srv_pp_tps"]) for r in rows if r.get("srv_pp_tps")]
    tg = [float(r["srv_tg_tps"]) for r in rows if r.get("srv_tg_tps")]
    if not tg:
        raise SystemExit(f"{a.csv} 里没有 srv_tg_tps，先看 metrics 有没有跑成")

    exists = os.path.exists(a.out) and os.path.getsize(a.out) > 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for test, vals in (("pp512", pp), ("tg128", tg)):
            if not vals:
                continue
            w.writerow(dict(quant=a.quant, file=a.file, ngl=a.ngl, test=test,
                            test_tps=f"{statistics.median(vals):.2f}",
                            stddev=f"{statistics.pstdev(vals) if len(vals) > 1 else 0.0:.2f}",
                            size_gib=a.size, commit="", host="wsl-t600"))
            print(f"  {a.quant:8s} {test:6s} median {statistics.median(vals):8.2f} t/s  (n={len(vals)})")


if __name__ == "__main__":
    main()
