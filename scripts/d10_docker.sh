#!/usr/bin/env bash
# scripts/d10_docker.sh — 备选路径：官方镜像（轮子里带全架构 PTX，能绕开「no kernel image」）
#
# 什么时候用：pip 路径报 KERNEL / ARCH（轮子没编 sm75）时。
# 什么时候别用：D11 之前，docker 的 GPU 透传本身没验证过，出问题分不清是镜像还是 runtime。
#
# 用法： bash scripts/d10_docker.sh --dry-run        # 只打印命令
#      IMG=vllm/vllm-openai:v0.8.5.post1 bash scripts/d10_docker.sh
set -uo pipefail
IMG=${IMG:-vllm/vllm-openai:v0.8.5.post1}
MODEL_DIR=${MODEL_DIR:-$HOME/models/qwen25-3b-awq}
PORT=${PORT:-8100}

CMD=(docker run --rm --gpus all
     -p "$PORT:8000"
     -v "$MODEL_DIR:/models/qwen25-3b-awq:ro"
     --ipc=host
     "$IMG"
     --model /models/qwen25-3b-awq
     --served-model-name qwen25-3b-awq
     --max-model-len 2048
     --gpu-memory-utilization 0.92
     --enforce-eager
     --max-num-seqs 2)

echo "== 前置检查 =="
docker info 2>/dev/null | grep -i -E "runtime|nvidia" | head -5 || echo "docker 不在或没权限"
echo "（若上面没有 nvidia，先装 nvidia-container-toolkit；WSL 下还要 Docker Desktop 勾 GPU 支持）"
echo

if [ "${1:-}" = "--dry-run" ]; then
  echo "${CMD[*]}"
  exit 0
fi

echo "${CMD[*]}"
"${CMD[@]}"
