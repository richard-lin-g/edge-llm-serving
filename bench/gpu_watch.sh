#!/usr/bin/env bash
# 显存轮询：另开一个终端跑，记录峰值
# 用法: ./bench/gpu_watch.sh [秒数，默认120] [采样间隔ms，默认500]
set -uo pipefail
SECS=${1:-120}
INTERVAL=${2:-500}
command -v nvidia-smi >/dev/null || { echo "[fail] 当前环境没有 nvidia-smi（确认在 WSL 里且 GPU 直通成功）"; exit 1; }

echo "[watch] ${SECS}s，每 ${INTERVAL}ms 采样 memory.used；Ctrl-C 提前结束"
timeout "$SECS" nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms "$INTERVAL" 2>/dev/null \
 | awk 'NF{if($1+0>m)m=$1+0; n++} END{printf "[peak] %d MiB  (%d samples)\n", m, n}'
