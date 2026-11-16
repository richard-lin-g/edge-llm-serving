#!/usr/bin/env bash
# D7 · 三档量化的裸速度（llama-bench，与 D2 同口径）
# 用法: bash scripts/d7_quant_bench.sh
#   NGL=36 bash scripts/d7_quant_bench.sh      # 默认 36（D2 实测最优）；跑不动自动退到 32/28/24
#   QUANTS="q4_k_m q8_0" bash scripts/d7_quant_bench.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BIN="${BIN:-$HOME/llama.cpp/build/bin}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
NGL="${NGL:-36}"
T="${T:-8}"
QUANTS="${QUANTS:-q4_0 q4_k_m q8_0}"
OUT="results/d7_quant_bench.csv"
COMMIT="${COMMIT:-$(git -C "$HOME/llama.cpp" log -1 --format=%h 2>/dev/null || echo unknown)}"

[ -x "$BIN/llama-bench" ] || { echo "缺 $BIN/llama-bench，先确认 D1/D2 的 CUDA 构建还在"; exit 1; }
mkdir -p results
: > results/d7_ngl.csv
echo "quant,ngl" >> results/d7_ngl.csv

for Q in $QUANTS; do
  M="$MODEL_DIR/qwen2.5-3b-instruct-${Q}.gguf"
  [ -s "$M" ] || { echo "! 缺 $M，跳过 $Q"; continue; }
  LOG="results/d7_bench_${Q}.log"

  N_OK=""
  TRY_LIST=$(printf '%s\n' "$NGL" 32 28 24 20 | awk -v n="$NGL" '$1<=n' | awk '!seen[$1]++')
  for TRY in $TRY_LIST; do
    echo "== [$Q] llama-bench -ngl $TRY -p 512 -n 128 -r 3 =="
    if "$BIN/llama-bench" -m "$M" -p 512 -n 128 -r 3 -ngl "$TRY" -t "$T" >"$LOG" 2>&1; then
      N_OK="$TRY"; break
    else
      echo "  -ngl $TRY 失败，末尾日志："
      tail -4 "$LOG" | sed 's/^/    /'
    fi
  done

  if [ -z "$N_OK" ]; then
    echo "! [$Q] 全部档位都跑不起来。要么显存不够（先关浏览器/桌面占用），要么模型文件坏了（sha 不对）"
    echo "$Q,none" >> results/d7_ngl.csv
    continue
  fi
  echo "$Q,$N_OK" >> results/d7_ngl.csv
  python3 bench/d7_parse_bench.py "$LOG" --quant "$Q" \
      --file "$(basename "$M")" --ngl "$N_OK" --commit "$COMMIT" --out "$OUT" || \
      echo "! [$Q] 解析失败，手工从 $LOG 里抄 pp512 / tg128"
done

echo
echo "== 汇总 =="
cat "$OUT" 2>/dev/null || echo "还没有结果"
echo "  注意：tg128 与服务器 srv_tg 相差约 10% 是正常的（D3 已记录），三档内部统一用 llama-bench 比，别混口径。"
