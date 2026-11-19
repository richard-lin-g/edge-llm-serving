#!/usr/bin/env bash
# scripts/d10_try.sh — 按「四级收紧」逐级尝试启动 vLLM，每一级都留完整证据
#
# 为什么是四级而不是一把梭：4 GB 卡上的失败点是渐变的——
#   L1 宽松：权重能进、KV 池也分到 block，但启动后期 OOM
#   L2 砍上下文到 1024：KV 需求减半
#   L3 砍到 768 + 只留 1 个序列：把「能起」和「能服务」分开验
#   L4 极限：512 ctx / 128 batched tokens，只为证明「这条路线到底有没有边界」
# 每一级都记：命令行、日志、显存采样、退出码。报告里要能一条条指过去。
#
# 用法：
#   source ~/venv-vllm/bin/activate
#   bash scripts/d10_try.sh 1          # 只跑 L1
#   bash scripts/d10_try.sh all        # 1→4，起得来就停
#   MODEL=~/models/qwen25-3b-awq LEVEL=2 bash scripts/d10_try.sh
set -uo pipefail

LEVEL=${LEVEL:-1}
MODEL=${MODEL:-$HOME/models/qwen25-3b-awq}
PORT=${PORT:-8100}
OUTDIR=${OUTDIR:-results}
WAIT_S=${WAIT_S:-300}
NAME=${NAME:-qwen25-3b-awq}
EXTRA=${EXTRA:-}          # 例：EXTRA="--attention-backend TRITON_ATTN"

mkdir -p "$OUTDIR"

run_level() {
  local lv=$1
  local ts log csv pid rc
  ts=$(date +%H%M%S)
  log="$OUTDIR/d10_server_L${lv}_${ts}.log"
  csv="$OUTDIR/d10_gpu_L${lv}_${ts}.csv"

  local args
  case "$lv" in
    1) args=(--max-model-len 2048 --gpu-memory-utilization 0.92 --enforce-eager
             --max-num-seqs 2 --max-num-batched-tokens 512) ;;
    2) args=(--max-model-len 1024 --gpu-memory-utilization 0.92 --enforce-eager
             --max-num-seqs 2 --max-num-batched-tokens 384) ;;
    3) args=(--max-model-len 768 --gpu-memory-utilization 0.94 --enforce-eager
             --max-num-seqs 1 --max-num-batched-tokens 256) ;;
    4) args=(--max-model-len 512 --gpu-memory-utilization 0.96 --enforce-eager
             --max-num-seqs 1 --max-num-batched-tokens 128) ;;
    *) echo "LEVEL 只能是 1..4 或 all"; return 2 ;;
  esac

  echo "================================================================"
  echo "L$lv  模型 $(basename "$MODEL")  port $PORT"
  echo "参数 ${args[*]} ${EXTRA}"
  echo "日志 $log ｜ 采样 $csv"
  echo "================================================================"

  echo "ts,util,mem_used_mib" > "$csv"
  (
    while true; do
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits \
        2>/dev/null | awk -v d="$(date +%H:%M:%S)" -F, '{gsub(/ /,"");print d","$1","$2}' >> "$csv"
      sleep 1
    done
  ) &
  local sampler=$!

  # 先把「空载基线」采 3 秒，报告里只报 峰值 − 基线
  sleep 3
  base=$(tail -n +2 "$csv" | cut -d, -f3 | sort -n | tail -1)
  echo "空载基线 ${base:-?} MiB"

  # shellcheck disable=SC2086
  python3 -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name "$NAME" \
      --host 127.0.0.1 --port "$PORT" \
      "${args[@]}" $EXTRA > "$log" 2>&1 &
  pid=$!

  local up=0 t=0
  while [ $t -lt "$WAIT_S" ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "进程已退出（${t}s）"
      break
    fi
    if python3 - "$PORT" <<'PY' >/dev/null 2>&1
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=2)
PY
    then up=1; echo "服务已就绪（${t}s）"; break; fi
    sleep 3; t=$((t+3))
  done

  if [ "$up" = "1" ]; then
    python3 bench/vllm_load.py --port "$PORT" --model "$NAME" --health || true
    echo "下一步： bash scripts/d10_load.sh $PORT"
  else
    echo "未就绪，收尾并判读日志"
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    rc=$?
    echo "退出码 ${rc:-unknown}"
  fi

  kill "$sampler" 2>/dev/null
  peak=$(tail -n +2 "$csv" | cut -d, -f3 | sort -n | tail -1)
  echo "显存：基线 ${base:-?} → 峰值 ${peak:-?} MiB（差 $(( ${peak:-0} - ${base:-0} )) MiB）"

  echo
  echo "-- 日志判读 --"
  python3 bench/vllm_classify.py "$log" | sed -n '1,40p'

  UP_LEVEL=$([ "$up" = "1" ] && echo "$lv" || echo "")
  return 0
}

if [ "$LEVEL" = "all" ]; then
  for lv in 1 2 3 4; do
    run_level "$lv"
    if [ -n "${UP_LEVEL:-}" ]; then
      echo
      echo ">>> L$lv 起起来了，停止继续收紧。先用 d10_load.sh 取数，再决定要不要再压极限。"
      exit 0
    fi
    echo
  done
  echo ">>> 四级全部未通过。这就是结论：本机在 4 GB 显存下无法启动 vLLM（证据：四份日志 + 四份显存采样）。"
else
  run_level "$LEVEL"
fi
