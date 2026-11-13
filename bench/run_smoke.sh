#!/usr/bin/env bash
# D1 冒烟：跑一次固定 prompt，落日志并抽取时间指标
# 用法: ./bench/run_smoke.sh <model.gguf> <ngl> [prompt]
set -uo pipefail
MODEL=${1:?用法: run_smoke.sh <model.gguf> <ngl> [prompt]}
NGL=${2:-99}
PROMPT=${3:-"你好，用两句话介绍你自己"}
CTX=${CTX:-4096}
NPRED=${NPRED:-64}
BIN=${LLAMA_CLI:-$HOME/llama.cpp/build/bin/llama-cli}

mkdir -p results
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="results/smoke-ngl${NGL}-${STAMP}.log"

echo "[run] model=$(basename "$MODEL") ngl=$NGL ctx=$CTX n=$NPRED bin=$BIN"
[ -x "$BIN" ] || { echo "[fail] 找不到可执行的 llama-cli: $BIN（检查编译或 LLAMA_CLI）"; exit 1; }
[ -f "$MODEL" ] || { echo "[fail] 模型文件不存在: $MODEL"; exit 1; }

WALL_START=$(date +%s%N)
"$BIN" -m "$MODEL" -ngl "$NGL" -c "$CTX" -n "$NPRED" --no-warmup -p "$PROMPT" </dev/null >"$LOG" 2>&1
RC=$?
WALL_MS=$(( ($(date +%s%N) - WALL_START) / 1000000 ))

echo "[log] $LOG  (exit=$RC, wall=${WALL_MS}ms)"
grep -E "load time|prompt eval time|eval time" "$LOG" || echo "[warn] 日志里没抓到时间行，手动看 $LOG"

if grep -qiE "failed to allocate|out of memory|CUDA error|cudaMalloc" "$LOG"; then
  echo "[oom] 本次分配失败 → 降 -ngl 重试（例如 99 → 32 → 24 → 16）"
fi
[ "$RC" -ne 0 ] && echo "[fail] llama-cli 退出码 $RC"
exit 0
