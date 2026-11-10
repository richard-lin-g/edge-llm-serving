#!/usr/bin/env bash
# D2 显存峰值测量：用 llama-bench 当负载（非交互、可复现），nvidia-smi 250ms 采样取稳态峰值
# 用法: scripts/vram_peak.sh <model.gguf> "0 10 20" [prompt_tokens]
# 注意: prompt_tokens 决定 KV 规模，测 4K 上下文用 4096，测 2K 上下文用 2048
set -u
MODEL="${1:?model path}"
NGLS="${2:?ngl list}"
PROMPT_TOKENS="${3:-4096}"
BENCH="${LLAMA_BENCH:-$HOME/llama.cpp/build/bin/llama-bench}"
mkdir -p results
echo "model,ngl,ctx,vram_peak_mib,rc"
for ngl in $NGLS; do
  SAMPLE=$(mktemp)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 250 >"$SAMPLE" 2>/dev/null &
  SMI=$!
  timeout 600 "$BENCH" -m "$MODEL" -ngl "$ngl" -p "$PROMPT_TOKENS" -n 128 -r 1 >/dev/null 2>&1
  RC=$?
  kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null
  PEAK=$(sort -n "$SAMPLE" | tail -1)
  echo "$(basename "$MODEL"),$ngl,$PROMPT_TOKENS,${PEAK:-0},$RC"
  rm -f "$SAMPLE"
done
