#!/usr/bin/env python3
"""bench/d7_parse_bench.py — 把 llama-bench 的表格输出解析成 CSV 一行两行

llama-bench 打印的是 markdown 表格，没有稳定机器格式，所以这里自己解：

  | model                  |    size |  params | backend | ngl |  test |    t/s |
  | llama 3B Q4_K - Medium | 1.79 GiB|  3.09 B | CUDA    |  36 | pp512 | 152.14 ± 1.2 |

用法：
  python3 bench/d7_parse_bench.py results/d7_bench_q4_k_m.log \
      --quant q4_k_m --file qwen2.5-3b-instruct-q4_k_m.gguf \
      --out results/d7_quant_bench.csv

退出码：解析到 pp512 且解析到 tg128 才算成功（少一个都算失败，避免“只有半个数”进报告）。
"""
import argparse
import csv
import os
import re
import sys

ROW = re.compile(r"\|\s*(pp\d+|tg\d+)\s*\|\s*([0-9.]+)\s*(?:±\s*([0-9.]+))?")
SIZE = re.compile(r"([0-9.]+)\s*GiB")
FIELDS = ["quant", "file", "size_gib", "ngl", "test", "test_tps", "stddev", "commit", "host"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--quant", required=True)
    ap.add_argument("--file", default="")
    ap.add_argument("--ngl", default="")
    ap.add_argument("--commit", default="")
    ap.add_argument("--host", default="")
    ap.add_argument("--out", default="results/d7_quant_bench.csv")
    a = ap.parse_args()

    text = open(a.log, encoding="utf-8", errors="replace").read()
    size = None
    m = SIZE.search(text)
    if m:
        size = float(m.group(1))

    found = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        m = ROW.search(line)
        if not m:
            continue
        test = m.group(1)
        found[test] = (float(m.group(2)), float(m.group(3) or 0.0))

    if not found:
        print(f"! {a.log} 里没找到 pp/tg 行——先看日志末尾有没有报错", file=sys.stderr)
        sys.exit(1)

    exists = os.path.exists(a.out) and os.path.getsize(a.out) > 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for test, (tps, sd) in sorted(found.items()):
            w.writerow(dict(quant=a.quant, file=a.file,
                            size_gib=f"{size:.2f}" if size else "",
                            ngl=a.ngl, test=test, test_tps=f"{tps:.2f}",
                            stddev=f"{sd:.2f}", commit=a.commit, host=a.host))
            print(f"  {a.quant:8s} {test:6s} {tps:8.2f} ± {sd:.2f} t/s"
                  f"{'  (' + format(size, '.2f') + ' GiB)' if size else ''}")

    if "tg128" not in found:
        print("! 只解析到 prefill，没有 tg128；这组数不能进三列表", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
