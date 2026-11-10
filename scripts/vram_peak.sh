#!/usr/bin/env bash
# 用法: scripts/vram_peak.sh <model.gguf> "0 10 20 28 32 36" [ctx]
set -u
MODEL="${1:?model path}"
NGLS="${2:?ngl list}"
CTX="${3:-4096}"
LLAMA="$HOME/llama.cpp/build/bin/llama-cli"
echo "model,ngl,ctx,vram_peak_mib,rc"
for ngl in $NGLS; do
  log=$(mktemp)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 250 >"$log" 2>/dev/null &
  smi=$!
  "$LLAMA" -m "$MODEL" -ngl "$ngl" -c "$CTX" -n 128 --no-warmup \
    -p "Explain why edge devices need quantized models in three sentences." \
    </dev/null >/tmp/run.log 2>&1
  rc=$?
  kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
  peak=$(sort -n "$log" | tail -1)
  echo "$(basename "$MODEL"),$ngl,$CTX,$peak,$rc"
  rm -f "$log"
done
