#!/usr/bin/env bash
# scripts/d10_load.sh — 对已起来的 vLLM 取一小份数（np 1/2，每档 20 笔）
#
# 口径纪律（和 D9 比只有一处必须改）：
#   精度档不同 —— vLLM 这边是 AWQ W4A16，llama.cpp 那边是 GGUF Q4_K_M。
#   所以这张表**不能和 d9_summary.csv 并排**，只能作为「同卡、同模型、不同引擎」的定性参考。
#   报告里必须写：两边的量化实现不同，吞吐数字不具可比性；可比的是「能不能起」与「调度行为」。
#
# 用法： bash scripts/d10_load.sh 8100
set -uo pipefail
PORT=${PORT:-8100}
MODEL=${MODEL_NAME:-qwen25-3b-awq}
NP_LIST=${NP_LIST:-1,2}
REQ=${REQ:-20}
MAXTOK=${MAXTOK:-64}

mkdir -p results
python3 bench/vllm_load.py --port "$PORT" --model "$MODEL" --health || exit 3

python3 bench/vllm_load.py \
  --port "$PORT" --model "$MODEL" \
  --np-list "$NP_LIST" --requests "$REQ" --max-tokens "$MAXTOK" \
  --tag d10 --out results/d10_requests.csv

echo
echo "看三件事就够："
echo "  1) 两档 n_out 是否恒定 64（不恒定 → ignore_eos 没生效，QPS 分母不齐）"
echo "  2) np2 的 P95 是否明显高于 np1（单序列卡上，第二个请求基本是排队）"
echo "  3) 有没有 fail（显存贴边时，第二个并发请求最容易失败）"
