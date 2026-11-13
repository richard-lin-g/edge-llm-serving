#!/usr/bin/env bash
# 用法: scripts/d3_run.sh a   # a=3B 全量(ngl36)  b=7B 台阶(22→20→18)
set -u
BIN="$HOME/llama.cpp/build/bin/llama-server"
M3B="$HOME/models/qwen2.5-3b-instruct-q4_k_m.gguf"
M7B="$HOME/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
PROFILE="${1:?用法: scripts/d3_run.sh a|b}"

if [ "$PROFILE" = "a" ]; then
  MODEL="$M3B"; CTX=4096; PORT=8081; NGLS="36"; PREFIX="a-3b"
else
  MODEL="$M7B"; CTX=2048; PORT=8082; NGLS="22 20 18"; PREFIX="b-7b"
fi

mkdir -p results logs
PID=""; NGL=""
for N in $NGLS; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  nohup "$BIN" -m "$MODEL" -ngl "$N" -c "$CTX" -np 1 \
        --host 127.0.0.1 --port "$PORT" > "logs/serve-$PREFIX-ngl$N.log" 2>&1 &
  PID=$!
  ready=0
  for i in $(seq 1 90); do
    if curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then ready=1; break; fi
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  if [ "$ready" = "1" ]; then NGL="$N"; break; fi
  echo "!! -ngl $N 起不来（或超时），降一档；报错首行："
  grep -m1 -E "error|out of memory|failed" "logs/serve-$PREFIX-ngl$N.log"
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; PID=""
done

if [ -z "$NGL" ]; then
  echo "三档都没起来 —— 今天按「7B 在 server 模式下不可用」记，写进 log"; exit 1
fi

TAG="$PREFIX-ngl$NGL"
echo "ready  port=$PORT  ngl=$NGL  ctx=$CTX  pid=$PID  tag=$TAG"
python3 bench/metrics.py --tag "$TAG" --base-url "http://127.0.0.1:$PORT" \
  --prompts bench/prompts.jsonl --repeat 3 --out results/d3_metrics.csv
echo "ngl=$NGL" > "logs/used-$TAG.txt"
kill "$PID"; sleep 3; nvidia-smi --query-gpu=memory.used --format=csv,noheader
