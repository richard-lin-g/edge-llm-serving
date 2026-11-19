#!/usr/bin/env bash
# D9 · 并发压测：np 1/4/8/16，每档 100 请求
#
# 关键口径：每槽上下文固定 1024，所以 -c = np × 1024。
# 不这么干的话，-np 16 -c 4096 会让每槽只剩 256，prompt 直接被截断，
# 四档曲线根本不是同一个实验。
#
# 用法:
#   bash scripts/d9_load.sh                       # 四档全跑
#   NP_LIST="4 8" bash scripts/d9_load.sh         # 只跑两档
#   CTX_SLOT=768 REQ=50 bash scripts/d9_load.sh   # 显存紧的时候降上下文/请求数
#   KEEP_SERVER=1 NP_LIST="16" bash scripts/d9_load.sh   # 跑完不关服务，方便手动 curl
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BIN="${BIN:-$HOME/llama.cpp/build/bin}"
MODEL="${MODEL:-$HOME/models/qwen2.5-3b-instruct-q4_k_m.gguf}"
NGL="${NGL:-36}"
T="${T:-8}"
B="${B:-512}"
UB="${UB:-256}"
CTX_SLOT="${CTX_SLOT:-1024}"      # 每槽上下文，四档必须一致
PORT="${PORT:-8092}"
NP_LIST="${NP_LIST:-1 4 8 16}"
REQ="${REQ:-100}"
OUT_TOKENS="${OUT_TOKENS:-64}"
WARMUP="${WARMUP:-4}"
RESET="${RESET:-1}"               # 1 = 每次重跑清空 CSV，避免旧档混进新表

[ -x "$BIN/llama-server" ] || { echo "! 缺 $BIN/llama-server"; exit 1; }
[ -s "$MODEL" ] || { echo "! 缺模型 $MODEL"; exit 1; }
command -v curl >/dev/null || { echo "! 没有 curl"; exit 1; }
command -v python3 >/dev/null || { echo "! 没有 python3"; exit 1; }
mkdir -p results

if [ "$RESET" = "1" ]; then
  rm -f results/d9_requests.csv results/d9_level_meta.csv results/d9_summary.csv
  rm -f results/d9_server_np*.log results/d9_baseline.csv
fi

vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
           | head -1 | tr -d '[:space:]'; }

baseline() {      # 空载基线取 5 次的最大值：桌面占用本身在抖，别拿最小值骗自己
  local mx=0 v
  for _ in 1 2 3 4 5; do
    v=$(vram); [ -n "${v:-}" ] && [ "$v" -gt "$mx" ] 2>/dev/null && mx=$v
    sleep 0.2
  done
  echo "${mx:-0}"
}

http_code() {      # 只取状态码：加载期间 /health 回 503，curl 退出码仍是 0，不能拿退出码当就绪
  curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null
}

free_port() {     # 上一档没退干净就起下一档，会撞 EADDRINUSE
  local code
  for _ in $(seq 1 30); do
    code=$(http_code)
    [ "$code" = "000" ] && return 0
    sleep 0.5
  done
  return 1
}

echo "== 先算天花板 =="
python3 bench/d9_ceiling.py --model "$MODEL" --np-list "$NP_LIST" || true

echo
echo "== 口径 =="
echo "  每档请求 $REQ 笔 / 热身 $WARMUP 笔 / 输出固定 $OUT_TOKENS token / 每槽 ctx $CTX_SLOT"
echo "  服务端: -ngl $NGL -t $T -b $B -ub $UB  端口 $PORT"
echo "  CSV: results/d9_requests.csv + results/d9_level_meta.csv"
echo

for NP in $NP_LIST; do
  C=$((NP * CTX_SLOT))
  LOG="results/d9_server_np${NP}.log"

  if ! free_port; then
    echo "! 端口 $PORT 还被占用，先 kill 掉残留的 llama-server（别 pkill -f，会误杀别的实验）"
    exit 1
  fi

  BASE=$(baseline)
  echo "== np=$NP  -c $C（每槽 $CTX_SLOT）  空载显存 ${BASE} MiB =="

  "$BIN/llama-server" -m "$MODEL" -ngl "$NGL" -c "$C" -np "$NP" \
      -b "$B" -ub "$UB" -t "$T" --port "$PORT" --no-warmup > "$LOG" 2>&1 &
  PID=$!

  READY=0
  for _ in $(seq 1 240); do
    if [ "$(http_code)" = "200" ]; then READY=1; break; fi
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done

  if [ "$READY" != "1" ]; then
    echo "  ! 服务没起来（大概率显存不够）。看 $LOG 末尾："
    tail -6 "$LOG" | sed 's/^/    /'
    echo "  处理：降 CTX_SLOT（1024→768）或 -b（2048→1024）后重跑这一档；"
    echo "        或直接记下「np=$NP 在本机起不来」，这也是结论（属于显存约束）。"
    kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
    sleep 2
    continue
  fi

  SLOTS=""; SLOTCTX=""
  for _ in $(seq 1 10); do
    SLOTS=$(grep -oE "n_slots = [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+")
    SLOTCTX=$(grep -oE "n_ctx_slot = [0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+")
    [ -n "${SLOTCTX:-}" ] && break
    sleep 1
  done
  echo "  日志: n_slots=${SLOTS:-?}  n_ctx_slot=${SLOTCTX:-?}"
  if [ -n "${SLOTCTX:-}" ] && [ "$SLOTCTX" != "$CTX_SLOT" ]; then
    echo "  ! 每槽上下文是 $SLOTCTX，不是我要的 $CTX_SLOT —— 四档不在同一口径，这档不采。"
    kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
    continue
  fi
  if [ -z "${SLOTCTX:-}" ]; then
    echo "  ! 日志里没找到 n_ctx_slot（版本改了日志格式？）手动确认后再决定要不要信这档数字"
  fi

  python3 bench/d9_load.py --port "$PORT" --np "$NP" --requests "$REQ" \
      --warmup "$WARMUP" --out-tokens "$OUT_TOKENS" \
      --out results/d9_requests.csv --meta results/d9_level_meta.csv \
      --c-ctx "$C" --ctx-slot "$CTX_SLOT" --tag d9
  RC=$?

  echo "$NP,$C,$CTX_SLOT,$BASE,$REQ,${SLOTS:-?},${SLOTCTX:-?},$RC" >> results/d9_baseline.csv

  if [ "${KEEP_SERVER:-0}" != "1" ]; then
    kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
    free_port || echo "  ! 端口没释放干净，下一档可能起不来"
  else
    echo "  KEEP_SERVER=1：服务留着（PID $PID），下一档会被端口挡住，记得手动 kill"
    break
  fi
done

echo
echo "== 汇总 =="
python3 bench/d9_summarize.py --requests results/d9_requests.csv \
    --meta results/d9_level_meta.csv --out results/d9_summary.csv --check
SUM_RC=$?

echo
echo "下一步："
echo "  1) 打开 docs/concurrency_模板.md，按 results/d9_summary.csv 填表（只填 CSV 里的数）"
echo "  2) 写 docs/log/d9.md（照 docs/log/d9_模板.md）"
echo "  3) bash scripts/d9_accept.sh"
exit "$SUM_RC"
