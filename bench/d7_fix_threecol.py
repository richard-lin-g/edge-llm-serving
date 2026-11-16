#!/usr/bin/env python3
"""把 summary 生成的三列表里那句 llama-bench 口径改成本次实际的口径（服务端自报）"""
import io

PATH = "results/d7_three_col.md"
OLD_PREFIX = "口径：llama-bench"
NEW = ("口径：服务端自报 srv_pp/srv_tg（bench/metrics.py，pp_n≈566、n_predict=128、"
       "每档 n=3 取中位数），固定 -ngl 36 / -t 8 / -b 512 / -c 4096；"
       "质量分为 10 题盲评 1–5 分均值（fact/format/useful 三项）。"
       "注：今日无 CUDA 版 llama-bench，数值不与 D2 的 llama-bench 39.33 同尺子，仅三档互比有效。")

with io.open(PATH, encoding="utf-8") as f:
    lines = f.read().splitlines()

hit = 0
for i, ln in enumerate(lines):
    if ln.startswith(OLD_PREFIX):
        lines[i] = NEW
        hit += 1

if hit == 0:
    raise SystemExit("没找到 llama-bench 口径行，先看一眼 results/d7_three_col.md")

with io.open(PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"已改 {hit} 行口径 → {PATH}")
