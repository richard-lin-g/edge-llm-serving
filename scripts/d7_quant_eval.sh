#!/usr/bin/env bash
# D7 · 三档依次起服务，采 10 题答案 + 记显存峰值，最后生成盲评稿
# 用法: bash scripts/d7_quant_eval.sh
#   QUANTS="q4_0 q4_k_m" bash scripts/d7_quant_eval.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BIN="${BIN:-$HOME/llama.cpp/build/bin}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
PORT="${PORT:-8081}"
T="${T:-8}"
QUANTS="${QUANTS:-q4_0 q4_k_m q8_0}"
ANSWERS="results/d7_answers_raw.csv"
VRAM="results/d7_vram.csv"

[ -x "$BIN/llama-server" ] || { echo "缺 $BIN/llama-server"; exit 1; }
mkdir -p results
[ -s "$ANSWERS" ] && cp "$ANSWERS" "$ANSWERS.bak.$(date +%H%M%S)"
printf 'quant,size_gib,ngl,vram_base_mib,vram_peak_mib\n' > "$VRAM"

ngl_of() {
  awk -F, -v q="$1" 'NR>1 && $1==q && $2!="none"{print $2}' results/d7_ngl.csv 2>/dev/null | tail -1
}

for Q in $QUANTS; do
  M="$MODEL_DIR/qwen2.5-3b-instruct-${Q}.gguf"
  [ -s "$M" ] || { echo "! 缺 $M，跳过"; continue; }
  N=$(ngl_of "$Q"); [ -z "$N" ] && N=36
  SZ=$(awk -v b="$(stat -c %s "$M")" 'BEGIN{printf "%.2f", b/1073741824}')
  echo "== [$Q] 起服务：-ngl $N -c 4096 -t $T（体积 $SZ GiB）"

  BASE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  [ -z "$BASE" ] && BASE=0

  "$BIN/llama-server" -m "$M" -ngl "$N" -c 4096 -t "$T" -b 512 \
      --host 127.0.0.1 --port "$PORT" > "results/d7_server_${Q}.log" 2>&1 &
  SPID=$!

  # 显存采样（服务在跑就每 300 ms 记一次）
  ( while kill -0 "$SPID" 2>/dev/null; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
      sleep 0.3
    done ) > "results/d7_vram_${Q}.txt" &
  MPID=$!

  READY=0
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then READY=1; break; fi
    kill -0 "$SPID" 2>/dev/null || break
    sleep 1
  done

  if [ "$READY" != 1 ]; then
    echo "! [$Q] 服务没起来，末尾日志："; tail -12 "results/d7_server_${Q}.log"
    kill "$SPID" "$MPID" 2>/dev/null; wait "$SPID" 2>/dev/null || true
    continue
  fi

  python3 bench/d7_eval.py collect --label "$Q" --base-url "http://127.0.0.1:$PORT" || \
    echo "! [$Q] 采答案失败（看上面报错）"

  kill "$MPID" 2>/dev/null; wait "$MPID" 2>/dev/null || true
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null || true

  PEAK=$(sort -n "results/d7_vram_${Q}.txt" 2>/dev/null | tail -1)
  [ -z "$PEAK" ] && PEAK=0
  echo "$Q,$SZ,$N,$BASE,$PEAK" >> "$VRAM"
  echo "  显存：基线 $BASE MiB → 峰值 $PEAK MiB（差 $((PEAK-BASE)) MiB）"
  sleep 2
done

echo
echo "== 显存 =="; cat "$VRAM"
echo
python3 bench/d7_eval.py check || { echo "! 答案不齐，检查哪一档没跑起来"; exit 1; }
python3 bench/d7_eval.py blind
echo
echo "下一步：打开 results/d7_blind.md，把 results/d7_scores.csv 的 fact/format/useful 填成 1–5。"
