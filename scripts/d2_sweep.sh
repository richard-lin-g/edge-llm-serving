#!/usr/bin/env bash
# D2 收尾测量：严格串行，避免并发污染显存峰值
# 峰值口径 C：llama-bench 当负载 + nvidia-smi 250ms 采样
set -u
cd "$HOME/edge-llm-serving" || exit 1
B="$HOME/llama.cpp/build/bin/llama-bench"
M7="$HOME/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
M3="$HOME/models/qwen2.5-3b-instruct-q4_k_m.gguf"
M8="$HOME/models/qwen2.5-3b-instruct-q8_0.gguf"

peak() {
  local s smi rc
  s=$(mktemp)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 250 >"$s" 2>/dev/null &
  smi=$!
  timeout 600 "$B" -m "$1" -ngl "$2" -p "$3" -n 128 -r 1 >/dev/null 2>&1
  rc=$?
  kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
  echo "$(basename "$1"),$2,$3,$(sort -n "$s" | tail -1),$rc"
  rm -f "$s"
}

echo "===BASELINE==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
date +%H:%M:%S

echo "===7B-CEILING==="
for n in 24 26 28; do
  echo "--ngl $n--"
  timeout 300 "$B" -m "$M7" -ngl "$n" -p 512 -n 128 -r 1 2>&1 | tail -4
done

echo "===7B-PEAK-NGL22==="
peak "$M7" 22 2048

echo "===3B-Q4KM-PEAK-SWEEP==="
echo "model,ngl,prompt_tokens,vram_peak_mib,rc"
for n in 0 10 12 16 20 28 32 36; do peak "$M3" "$n" 4096; done

echo "===Q8-CHECK==="
SIZE=$(stat -c %s "$M8" 2>/dev/null || echo 0)
echo "q8_bytes=$SIZE"
if [ "$SIZE" -ge 3616088480 ]; then
  echo "===Q8-SPEED==="
  for n in 28 32 36; do
    echo "--q8 ngl $n--"
    timeout 300 "$B" -m "$M8" -ngl "$n" -p 512 -n 128 -r 3 2>&1 | tail -4
  done
  echo "===Q8-PEAK==="
  echo "model,ngl,prompt_tokens,vram_peak_mib,rc"
  for n in 28 32 36; do peak "$M8" "$n" 4096; done
else
  echo "Q8_NOT_READY"
fi

echo "===ALLDONE==="
date +%H:%M:%S
