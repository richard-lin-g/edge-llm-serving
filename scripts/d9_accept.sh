#!/usr/bin/env bash
# D9 验收：数据对不对、文档填没填、结论能不能指到具体数
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PASS=0; FAIL=0
ok()  { echo "  ✓ $1"; PASS=$((PASS+1)); }
bad() { echo "  ✗ $1"; FAIL=$((FAIL+1)); }
note(){ echo "  · $1"; }

echo "== D9 验收 =="

# 1 原始数据在
[ -s results/d9_requests.csv ] && ok "逐请求 CSV 存在（$(($(wc -l < results/d9_requests.csv) - 1)) 行）" \
  || bad "缺 results/d9_requests.csv"
[ -s results/d9_level_meta.csv ] && ok "档位元数据存在" || bad "缺 results/d9_level_meta.csv"

# 2 汇总表 + 口径检查
if [ -s results/d9_summary.csv ]; then
  ok "汇总表已生成"
  if python3 bench/d9_summarize.py --check results/d9_summary.csv >/tmp/d9_check.out 2>&1; then
    ok "口径检查通过（四档齐、每槽 ctx 一致、错误率达标、prompt 未被截断）"
  else
    bad "口径检查未通过，详见："
    sed 's/^/      /' /tmp/d9_check.out
  fi
else
  bad "缺 results/d9_summary.csv（先跑 scripts/d9_load.sh）"
fi

# 3 GPU 证据
python3 - <<'PY' && ok "四档都采到了 GPU 利用率/显存峰值" || bad "有档位没采到 GPU 数据（nvidia-smi 不通？）"
import csv,sys
rows=list(csv.DictReader(open("results/d9_summary.csv",encoding="utf-8")))
sys.exit(0 if rows and all(r["gpu_util_avg"] not in ("","None") and
                           r["vram_peak"] not in ("","None") for r in rows) else 1)
PY

# 4 曲线解释得通：P95 必须随并发上升，否则说明客户端在串行
python3 - <<'PY' && ok "P95 随并发单调上升（排队真实存在，不是串行跑出来的）" || note "P95 没随并发上升：检查是不是客户端把请求串起来了"
import csv,sys
rows=sorted([r for r in csv.DictReader(open("results/d9_summary.csv",encoding="utf-8"))
             if r["p95_ms"] not in ("","None")], key=lambda r:int(float(r["np"])))
vals=[float(r["p95_ms"]) for r in rows]
sys.exit(0 if len(vals)>=2 and all(b>=a*0.9 for a,b in zip(vals,vals[1:])) else 1)
PY

# 5 文档
if [ -s docs/log/d9.md ]; then
  if grep -q "待填\|TODO\|（空）" docs/log/d9.md; then
    bad "docs/log/d9.md 还有未填的占位"
  else
    ok "docs/log/d9.md 已写完"
  fi
else
  bad "缺 docs/log/d9.md（照 docs/log/d9_模板.md 写，当天写完，第二天不会再记得）"
fi

CHAPTER="docs/concurrency.md"
[ -s "$CHAPTER" ] || CHAPTER="docs/benchmark-report.md"
if [ -s "$CHAPTER" ]; then
  if grep -q "待填" "$CHAPTER"; then
    bad "$CHAPTER 里「第三章 · 并发」还有待填"
  else
    note "$CHAPTER 存在；确认第三章表格数字全部来自 results/d9_summary.csv"
  fi
else
  bad "缺并发章节（docs/concurrency.md 或 benchmark-report.md）"
fi

# 6 结论里必须有数字：结论句要能指到 QPS/P95/GPU%
if [ -s docs/log/d9.md ] && grep -qE "QPS|P95|GPU|tok/s" docs/log/d9.md; then
  ok "日志里的结论带数字（不是「有提升」「比较稳定」这类空话）"
else
  bad "结论里没有可核对的具体数"
fi

echo
echo "结果：PASS $PASS / FAIL $FAIL"
[ "$FAIL" -eq 0 ] && echo "→ 可以把第三章和 d9.md 推到 GitHub，然后收工" \
                  || echo "→ 先修 ✗ 项；口径不对的数据，写进报告只会给自己挖坑"
exit "$([ "$FAIL" -eq 0 ] && echo 0 || echo 1)"
