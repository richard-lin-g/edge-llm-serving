#!/usr/bin/env bash
# scripts/d10_accept.sh — D10 收工验收，全绿才提交
# 用法： bash scripts/d10_accept.sh
# 说明：D10 的验收重点不是"跑出数字"，而是"结论有据"——起不来也必须有证据链。
set -uo pipefail

PASS=0; FAIL=0
ok()  { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

has()    { local f=$1 pat=$2 label=$3; if grep -qE "$pat" "$f" 2>/dev/null; then ok "$label"; else bad "$label"; fi; }
nothas() { local f=$1 pat=$2 label=$3; if ! grep -qE "$pat" "$f" 2>/dev/null; then ok "$label"; else bad "$label（不该出现：$pat）"; fi; }
runok()  { local label=$1; shift; if "$@" >/dev/null 2>&1; then ok "$label"; else bad "$label"; fi; }

echo "== D10 验收 =="

runok "01 mem_budget 算式自检" python3 bench/mem_budget.py --check
if [ -s results/d10_vram.json ]; then ok "02 显存探针结果已落盘 results/d10_vram.json"; else bad "02 缺 results/d10_vram.json（跑 python3 bench/vram_probe.py --json > results/d10_vram.json）"; fi

LOG=$(ls -t results/d10_server_*.log 2>/dev/null | head -1)
if [ -n "${LOG:-}" ]; then
  ok "03 有启动日志：$(basename "$LOG")（$(wc -l < "$LOG") 行）"
else
  bad "03 没有 results/d10_server_*.log：一次启动都没试"
fi

if [ -n "${LOG:-}" ]; then
  CODE=$(python3 bench/vllm_classify.py "$LOG" --json 2>/dev/null \
         | python3 -c "import sys,json;print(json.load(sys.stdin)[0]['verdict']['code'])" 2>/dev/null || echo UNKNOWN)
  case "${CODE:-UNKNOWN}" in
    NO_ERROR) ok "04 启动日志无报错（视为已起，需 CSV 佐证）" ;;
    UNKNOWN)  bad "04 日志判定 UNKNOWN：贴末尾 40 行再定，不许带不明结论提交" ;;
    *)        ok "04 失败已归因：$CODE" ;;
  esac
fi

DOC=docs/d10_vllm.md
if [ -f "$DOC" ]; then
  ok "05 $DOC 存在"
  nothas "$DOC" '\{\{|TODO|待填' "06 报告无未填占位符"
  has    "$DOC" '实测可用|可用显存' "07 报告写了实测可用显存"
  has    "$DOC" 'vLLM ?[0-9]+\.[0-9]+|版本[：:] ?[0-9]+\.[0-9]+' "08 报告写了 vLLM 版本号"
else
  bad "05 $DOC 不存在（从 docs/d10_vllm_模板.md 复制）"
fi

if [ -f results/d10_requests.csv ]; then
  ok "09 有实测 CSV（须标注与本机 llama.cpp 口径不可比）"
  DUP=$(python3 - <<'PY' 2>/dev/null
import csv, collections
rows = list(csv.DictReader(open('results/d10_requests.csv')))
bad = [np_ for np_ in {r['np'] for r in rows}
       if len({r['n_out'] for r in rows if r['np'] == np_}) > 1]
print(",".join(bad))
PY
)
  if [ -z "${DUP:-}" ]; then ok "10 每档 n_out 恒定"; else bad "10 这些档 n_out 不恒定：$DUP"; fi
else
  if grep -qE '未取得实测|没有吞吐数据|未进入服务阶段' "$DOC" 2>/dev/null; then
    ok "09 无 CSV，但报告已明确声明「未取得实测数据」"
  else
    bad "09 无 CSV，也没有写清「未取得实测数据」——这就是编数据的开端"
  fi
  ok "10 （无 CSV，跳过 n_out 检查）"
fi

has "$DOC" 'W =|36 KiB/token|权重.*KV.*运行时' "11 报告含三笔账（权重/KV/运行时）算式"
has "docs/log/d10.md" '明天|D11' "12 docs/log/d10.md 含明日清单"
has "$DOC" 'PagedAttention' "13 报告讲了 PagedAttention"
has "$DOC" 'continuous batching|连续批处理' "14 报告讲了连续批处理"

echo
echo "== 结果：PASS $PASS / FAIL $FAIL =="
if [ "$FAIL" -eq 0 ]; then
  echo "可以提交： git add -A && git commit -m 'D10: vLLM feasibility + constraint analysis' && git push"
else
  echo "先补齐失败项再提交；D10 宁可空着，也不留一句没数支撑的结论。"
fi
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
