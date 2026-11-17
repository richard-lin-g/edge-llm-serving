#!/usr/bin/env python3
"""bench/d8_parse_bench.py — 解析 llama-bench 的 markdown 表格，抽 pp/tg（含 @ d 深度）

本机 llama-bench（build 785322d）用 `-d N` 指定 KV 深度，行标签形如：
  | ... | pp512 @ d2048 | 126.70 ± 0.00 |
  | ... | tg64  @ d2048 |  37.54 ± 0.00 |

用法：
  python3 bench/d8_parse_bench.py results/d8_depth_2048.log --depth 2048 \\
      --ngl 36 --out results/d8_depth.csv
"""
import argparse
import csv
import os
import re
import sys

ROW = re.compile(r"\|\s*(pp\d+|tg\d+)\s*(?:@\s*d(\d+))?\s*\|\s*([0-9.]+)\s*(?:±\s*([0-9.]+))?")
FIELDS = ["depth", "ngl", "pp_tps", "pp_std", "tg_tps", "tg_std", "commit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--depth", default="")
    ap.add_argument("--ngl", default="")
    ap.add_argument("--commit", default="")
    ap.add_argument("--out", default="results/d8_depth.csv")
    a = ap.parse_args()

    text = open(a.log, encoding="utf-8", errors="replace").read()
    found = {}
    depth_in_row = ""
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        m = ROW.search(line)
        if not m:
            continue
        kind = m.group(1)[:2]
        found[kind] = (float(m.group(3)), float(m.group(4) or 0.0))
        if m.group(2):
            depth_in_row = m.group(2)

    if "pp" not in found or "tg" not in found:
        print(f"! {a.log} 没同时拿到 pp 与 tg；看日志末尾是否 OOM", file=sys.stderr)
        return 1

    depth = a.depth or depth_in_row
    pp_t, pp_s = found["pp"]
    tg_t, tg_s = found["tg"]

    exists = os.path.exists(a.out) and os.path.getsize(a.out) > 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(dict(depth=depth, ngl=a.ngl,
                        pp_tps=f"{pp_t:.2f}", pp_std=f"{pp_s:.2f}",
                        tg_tps=f"{tg_t:.2f}", tg_std=f"{tg_s:.2f}",
                        commit=a.commit))
    print(f"  depth={depth:>5}  pp {pp_t:8.2f} ± {pp_s:.2f}   tg {tg_t:7.2f} ± {tg_s:.2f} t/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
