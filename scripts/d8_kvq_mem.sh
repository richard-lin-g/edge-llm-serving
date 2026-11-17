#!/bin/bash
# D8 · KV 量化在「静态占用」口径下的显存收益（llama-server，-np 1，ctx 8192）
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
BIN="${BIN:-$HOME/llama.cpp/build/bin}"
MODEL="${MODEL:-$HOME/models/qwen2.5-3b-instruct-q4_k_m.gguf}"
CTX="${CTX:-8192}"
PORT="${PORT:-8092}"
OUT="results/d8_kvq_mem.csv"
echo "ctk,ctv,ctx,vram_base_mib,vram_peak_mib,delta_mib" > "$OUT"

vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]'; }

run() {
  local CTK=$1 CTV=$2
  local LOG="results/d8_kvqmem_${CTK}_${CTV}.log"
  local BASE PEAK V
  BASE=$(vram)
  echo "== ctk=$CTK ctv=$CTV ctx=$CTX  基线 ${BASE} MiB =="
  "$BIN/llama-server" -m "$MODEL" -ngl 36 -c "$CTX" -np 1 -t 8 \
      -ctk "$CTK" -ctv "$CTV" --port "$PORT" --no-warmup > "$LOG" 2>&1 &
  local PID=$!
  PEAK=$BASE
  for _ in $(seq 1 120); do
    V=$(vram); [ -n "${V:-}" ] && [ "$V" -gt "$PEAK" ] 2>/dev/null && PEAK=$V
    kill -0 "$PID" 2>/dev/null || break
    curl -s -o /dev/null "http://127.0.0.1:${PORT}/health" 2>/dev/null && break
    sleep 0.5
  done
  for _ in $(seq 1 20); do
    V=$(vram); [ -n "${V:-}" ] && [ "$V" -gt "$PEAK" ] 2>/dev/null && PEAK=$V
    sleep 0.15
  done
  echo "  峰值 ${PEAK} MiB（+$((PEAK-BASE))）"
  echo "$CTK,$CTV,$CTX,$BASE,$PEAK,$((PEAK-BASE))" >> "$OUT"
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; sleep 3
}

run f16  f16
run q8_0 f16
run q8_0 q8_0
echo; cat "$OUT"
