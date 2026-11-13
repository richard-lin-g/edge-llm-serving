#!/usr/bin/env bash
# D2 卸载层数扫描：-ngl 逐档跑，输出 results/ngl_sweep.csv
# 用法: LEVELS="0 8 16 24 32 99" ./bench/ngl_sweep.sh <model.gguf>
set -uo pipefail
MODEL=${1:?用法: ngl_sweep.sh <model.gguf>}
LEVELS=${LEVELS:-"0 8 16 24 32 99"}
CTX=${CTX:-4096}
NPRED=${NPRED:-64}
PROMPT=${PROMPT:-"你好，用两句话介绍你自己"}
BIN=${LLAMA_CLI:-$HOME/llama.cpp/build/bin/llama-cli}
OUT_CSV=${OUT_CSV:-results/ngl_sweep.csv}

mkdir -p results
[ -f "$OUT_CSV" ] || echo "model,ngl,ctx,n_predict,tokens_per_s,load_ms,oom,stamp" > "$OUT_CSV"

for L in $LEVELS; do
  STAMP=$(date +%Y%m%d-%H%M%S)
  LOG="results/sweep-ngl${L}-${STAMP}.log"
  echo "==> -ngl $L"
  "$BIN" -m "$MODEL" -ngl "$L" -c "$CTX" -n "$NPRED" --no-warmup -p "$PROMPT" </dev/null >"$LOG" 2>&1

  TPS=$(grep -E "eval time" "$LOG" | tail -1 | grep -oE "[0-9.]+ tokens per second" | grep -oE "^[0-9.]+" || true)
  LOAD=$(grep -oE "load time = *[0-9.]+" "$LOG" | head -1 | grep -oE "[0-9.]+" || true)
  OOM=0
  grep -qiE "failed to allocate|out of memory|CUDA error|cudaMalloc" "$LOG" && OOM=1
  [ -z "${TPS:-}" ] && TPS=""
  [ -z "${LOAD:-}" ] && LOAD=""

  echo "$(basename "$MODEL"),$L,$CTX,$NPRED,$TPS,$LOAD,$OOM,$STAMP" >> "$OUT_CSV"
  echo "    tokens/s=${TPS:-NA}  load=${LOAD:-NA}ms  oom=$OOM"
done

echo "[done] $OUT_CSV"
column -s, -t "$OUT_CSV" 2>/dev/null || cat "$OUT_CSV"
