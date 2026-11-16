#!/usr/bin/env bash
# D7 · 收工验收（14 项）。退出码 0 = 今天可以提交
# 用法: bash scripts/d7_accept.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
PASS=0; FAIL=0
ck(){ local d="$1"; shift
  if "$@" >/dev/null 2>&1; then printf '  [PASS] %s\n' "$d"; PASS=$((PASS+1));
  else printf '  [FAIL] %s\n' "$d"; FAIL=$((FAIL+1)); fi; }

# --- 小工具：把判断交给 python，避免 shell 转义踩坑 ---
big_enough(){ [ -s "$1" ] && [ "$(stat -c %s "$1")" -ge "$2" ]; }

bench_rows(){
python3 - <<'PY'
import csv,sys
rows=list(csv.DictReader(open('results/d7_quant_bench.csv',encoding='utf-8')))
tests={(r['quant'],r['test']) for r in rows}
ok = len(rows)>=6 and len({r['quant'] for r in rows})>=3 and ('q4_k_m','tg128') in tests
sys.exit(0 if ok else 1)
PY
}

anchor_ok(){
python3 - <<'PY'
import csv,sys
rows=list(csv.DictReader(open('results/d7_quant_bench.csv',encoding='utf-8')))
tg=[float(r['test_tps']) for r in rows if r['quant']=='q4_k_m' and r['test']=='tg128']
if not tg: sys.exit(1)
d=abs(tg[0]-39.33)/39.33
sys.exit(0 if d<=0.10 else 1)
PY
}

answers_ok(){
python3 - <<'PY'
import csv,sys
rows=list(csv.DictReader(open('results/d7_answers_raw.csv',encoding='utf-8')))
q=len({r['qid'] for r in rows}); qt=len({r['quant'] for r in rows})
sys.exit(0 if q>=10 and qt>=3 and len(rows)>=30 else 1)
PY
}

scores_ok(){
python3 - <<'PY'
import csv,sys
rows=list(csv.DictReader(open('results/d7_scores.csv',encoding='utf-8')))
def bad(v):
    v=(v or '').strip()
    return not (v.isdigit() and 1<=int(v)<=5)
n=sum(1 for r in rows if not any(bad(r.get(k)) for k in ('fact','format','useful')))
sys.exit(0 if len(rows)>=30 and n==len(rows) else 1)
PY
}

quality_ok(){
python3 - <<'PY'
import csv,sys
rows=list(csv.DictReader(open('results/d7_quality.csv',encoding='utf-8')))
sys.exit(0 if len({r['quant'] for r in rows})>=3 else 1)
PY
}

vram_ok(){
python3 - <<'PY'
import csv,sys
rows=[r for r in csv.DictReader(open('results/d7_vram.csv',encoding='utf-8'))
      if (r.get('vram_peak_mib') or '0') not in ('0','')]
sys.exit(0 if len(rows)>=3 else 1)
PY
}

echo "== D7 验收：量化三列 =="
echo "-- 模型 --"
ck "Q4_0 ≥1.6 GB"   big_enough "$MODEL_DIR/qwen2.5-3b-instruct-q4_0.gguf" 1600000000
ck "Q4_K_M ≥1.9 GB" big_enough "$MODEL_DIR/qwen2.5-3b-instruct-q4_k_m.gguf" 1900000000
ck "Q8_0 ≥3.0 GB"   big_enough "$MODEL_DIR/qwen2.5-3b-instruct-q8_0.gguf" 3000000000

echo "-- 速度 --"
ck "bench 6 行、覆盖 3 档、含 Q4_K_M tg128" bench_rows
ck "Q4_K_M tg128 与 D2 锚点偏差 ≤10%" anchor_ok

echo "-- 质量 --"
ck "答案 3 档 × 10 题齐全" answers_ok
ck "盲评稿含 A/B/C" bash -c "grep -q '答案 A' results/d7_blind.md && grep -q '答案 C' results/d7_blind.md"
ck "打分表 30 条全部填了 1-5" scores_ok
ck "解盲后 3 档都有分" quality_ok
ck "显存 3 档都有峰值" vram_ok

echo "-- 文档 --"
ck "三列对照表已生成" bash -c "grep -q 'q4_k_m' results/d7_three_col.md"
ck "quant-comparison 有结论且无 TODO" bash -c "[ -s docs/quant-comparison.md ] && grep -q '## .*结论' docs/quant-comparison.md && ! grep -q 'TODO' docs/quant-comparison.md"
ck "d7.md 无 TODO" bash -c "[ -s docs/log/d7.md ] && ! grep -q 'TODO' docs/log/d7.md"
ck "提交数 ≥6" bash -c "test \$(git rev-list --count HEAD 2>/dev/null || echo 0) -ge 6"

echo
echo "PASS=$PASS FAIL=$FAIL"
if [ "$FAIL" = 0 ]; then echo "全绿：可以 commit + push"; exit 0; fi
echo "还有 $FAIL 项没过，逐条对着上面修"; exit 1
