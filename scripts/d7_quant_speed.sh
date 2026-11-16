#!/usr/bin/env bash
# D7 · 三档速度：服务端口径（srv_pp / srv_tg）
# 为什么不用 llama-bench：这台机器的 llama-bench 是 Debian CPU-only 包，没有 CUDA 后端；
# 而 repo/bin/llama-server 是 CUDA 版。为了三档同尺子，改用量具 bench/metrics.py 的服务端自报值。
# 与 D2 的 39.33 差异：srv_tg 与 llama-bench tg128 基本同口径（D3 已验证），可直接对表。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BIN="${BIN:-$HOME/edge-llm-serving/bin}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
PORT="${PORT:-8081}"
QUANTS="${QUANTS:-q4_0 q4_k_m q8_0}"
OUT="results/d7_quant_bench.csv"
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
mkdir -p results

# 速度专用 prompt：只取 L 档，repeat 加到 pp_n ≈ 512
if [ ! -s bench/prompts_d7_speed.jsonl ]; then
python3 - <<'PY'
import json
src = [json.loads(l) for l in open('bench/prompts.jsonl', encoding='utf-8') if l.strip()]
pick = [p for p in src if p['id'] == 'l1'][0]
pick = dict(pick, id='spd', tier='L', repeat=5)
with open('bench/prompts_d7_speed.jsonl', 'w', encoding='utf-8') as f:
    f.write(json.dumps(pick, ensure_ascii=False) + '\n')
print('生成 bench/prompts_d7_speed.jsonl（l1 ×5 段）')
PY
fi

: > results/d7_ngl.csv
echo "quant,ngl" >> results/d7_ngl.csv

try_server() {  # $1 model  $2 ngl  -> 0 可用 / 1 不可用
  local m="$1"
  local n="$2"
  local log="results/d7_probe_$(basename "$m" .gguf)_$n.log"
  "$BIN/llama-server" -m "$m" -ngl "$n" -c 4096 -t 8 -b 512 \
      --host 127.0.0.1 --port "$PORT" >"$log" 2>&1 &
  local pid=$!
  local ready=0
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if [ "$ready" != 1 ]; then
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    echo "    -ngl $n 起不来（$(tail -2 "$log" | tr '\n' ' ' | cut -c1-120)）"
    return 1
  fi
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/completions" \
      -H 'Content-Type: application/json' \
      -d '{"prompt":"ping","n_predict":4,"temperature":0,"stream":false}')
  if [ "$code" != 200 ]; then
    echo "    -ngl $n 起得来但首请求 HTTP $code（多半 CUDA OOM）"
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    return 1
  fi
  echo "    -ngl $n 可用"
  return 0
}

for Q in $QUANTS; do
  M="$MODEL_DIR/qwen2.5-3b-instruct-${Q}.gguf"
  [ -s "$M" ] || { echo "! 缺 $M，跳过 $Q"; continue; }
  SZ=$(awk -v b="$(stat -c %s "$M")" 'BEGIN{printf "%.2f", b/1073741824}')
  echo "== [$Q] $SZ GiB =="

  N_OK=""
  for TRY in 36 32 28 24 20; do
    if try_server "$M" "$TRY"; then N_OK="$TRY"; break; fi
  done
  if [ -z "$N_OK" ]; then
    echo "! [$Q] 全部档位起不来，跳过（在报告里标注未取得）"
    echo "$Q,none" >> results/d7_ngl.csv
    continue
  fi
  echo "$Q,$N_OK" >> results/d7_ngl.csv

  mkdir -p results
  python3 bench/metrics.py --tag "d7-${Q}-ngl${N_OK}" --base-url "http://127.0.0.1:$PORT" \
      --prompts bench/prompts_d7_speed.jsonl --n-predict 128 --repeat 3 \
      --out "results/d7_speed_${Q}.csv"
  RC=$?

  # 关掉这次起的服务（try_server 已在后台、这里按端口收）
  pkill -f "llama-server .*-port $PORT" 2>/dev/null
  sleep 3

  if [ "$RC" = 0 ]; then
    python3 bench/d7_speed_to_bench.py --csv "results/d7_speed_${Q}.csv" --quant "$Q" \
        --ngl "$N_OK" --size "$SZ" --file "$(basename "$M")" --out "$OUT"
  else
    echo "! [$Q] metrics 返回 $RC，本档速度不完整"
  fi
done

echo
echo "== 结果 =="
[ -s "$OUT" ] && cat "$OUT" || echo "没有产出"
echo "注：test 列 pp512/tg128 是服务端 srv_pp/srv_tg 的中位数，不是 llama-bench 直方。"
