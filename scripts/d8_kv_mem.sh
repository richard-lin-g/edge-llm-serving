#!/bin/bash
# D8 · KV Cache 静态占用：ctx 2K/4K/8K 的实测显存 vs 公式
# 采集端用 llama-server（-np 1，单槽），因为本机 llama-cli 在此构建下无输出。
#
# 用法: bash scripts/d8_kv_mem.sh
#       CTX_LIST="2048 4096" bash scripts/d8_kv_mem.sh
#       NGL=32 bash scripts/d8_kv_mem.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BIN="${BIN:-$HOME/llama.cpp/build/bin}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL="${MODEL:-$MODEL_DIR/qwen2.5-3b-instruct-q4_k_m.gguf}"
NGL="${NGL:-36}"
T="${T:-8}"
CTX_LIST="${CTX_LIST:-2048 4096 8192}"
PORT="${PORT:-8091}"
OUT="results/d8_kv_mem.csv"

[ -x "$BIN/llama-server" ] || { echo "! 缺 $BIN/llama-server"; exit 1; }
[ -s "$MODEL" ] || { echo "! 缺模型 $MODEL"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "! 没有 nvidia-smi"; exit 1; }

mkdir -p results
python3 bench/d8_gguf_meta.py "$MODEL" --out results/d8_kv_model.json || exit 1

echo "ctx,ngl,threads,slots,kv_formula_mib,kv_log_mib,vram_base_mib,vram_peak_mib,notes" > "$OUT"

vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]'; }

baseline() {
  local mx=0 v
  for i in 1 2 3 4 5; do
    v=$(vram); [ -n "${v:-}" ] && [ "$v" -gt "$mx" ] 2>/dev/null && mx=$v
    sleep 0.2
  done
  echo "$mx"
}

wait_ready() {   # 最多等 90 秒
  local i=0
  while [ $i -lt 90 ]; do
    if curl -s -o /dev/null "http://127.0.0.1:${PORT}/health" 2>/dev/null; then return 0; fi
    sleep 1; i=$((i+1))
  done
  return 1
}

for CTX in $CTX_LIST; do
  LOG="results/d8_kv_ctx${CTX}.log"
  BASE=$(baseline)
  FML=$(python3 bench/d8_kv_formula.py --model results/d8_kv_model.json --ctx "$CTX" --mib)

  echo "== ctx=$CTX  公式 ${FML} MiB  基线 ${BASE} MiB =="
  "$BIN/llama-server" -m "$MODEL" -ngl "$NGL" -c "$CTX" -np 1 -t "$T" \
      --port "$PORT" --no-warmup > "$LOG" 2>&1 &
  PID=$!

  PEAK=$BASE
  # 加载期间连续采样
  for _ in $(seq 1 120); do
    V=$(vram); [ -n "${V:-}" ] && [ "$V" -gt "$PEAK" ] 2>/dev/null && PEAK=$V
    kill -0 "$PID" 2>/dev/null || break
    if curl -s -o /dev/null "http://127.0.0.1:${PORT}/health" 2>/dev/null; then break; fi
    sleep 0.5
  done

  if ! wait_ready; then
    echo "  ! 服务 90 秒内没起来，看 $LOG 末尾"
    tail -5 "$LOG" | sed 's/^/    /'
    echo "$CTX,$NGL,$T,1,$FML,na,$BASE,na,server_not_ready" >> "$OUT"
    kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; sleep 2
    continue
  fi

  # 就绪后再采 3 秒，取峰值
  for _ in $(seq 1 20); do
    V=$(vram); [ -n "${V:-}" ] && [ "$V" -gt "$PEAK" ] 2>/dev/null && PEAK=$V
    sleep 0.15
  done

  SLOTS=$(grep -oE "n_slots = [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+")
  SLOTCTX=$(grep -oE "n_ctx_slot = [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+")
  KVLOG=$(grep -iE "kv .*size" "$LOG" 2>/dev/null | grep -oE "[0-9]+\.?[0-9]* *MiB" | awk '{print $1}' | sort -g | tail -1)
  KVLOG=${KVLOG:-na}

  echo "  显存峰值 ${PEAK} MiB（基线 ${BASE}）  日志KV ${KVLOG} MiB  slots=${SLOTS:-?} slot_ctx=${SLOTCTX:-?}"
  echo "$CTX,$NGL,$T,${SLOTS:-1},$FML,$KVLOG,$BASE,$PEAK,ok" >> "$OUT"

  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  sleep 3
done

echo
echo "== 对表（差值才是纯 KV 增量）=="
python3 bench/d8_kv_formula.py --model results/d8_kv_model.json \
    --check "$OUT" --out results/d8_kv_mem_checked.csv
