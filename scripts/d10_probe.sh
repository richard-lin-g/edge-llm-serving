#!/usr/bin/env bash
# scripts/d10_probe.sh — 开工第一件事：一屏看清 驱动 / 显存 / CUDA / docker / 模型在不在
# 用法： bash scripts/d10_probe.sh
set -uo pipefail

echo "== 时间 =="
date '+%F %T %a'

echo
echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap \
             --format=csv
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
else
  echo "nvidia-smi 不在 PATH：WSL 的 /usr/lib/wsl/lib 没进 PATH？"
  echo "试： ls /usr/lib/wsl/lib/nvidia-smi && export PATH=\$PATH:/usr/lib/wsl/lib"
fi

echo
echo "== CUDA / Python =="
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -2 || echo "nvcc: 未安装（vLLM 不需要，编译 llama.cpp 需要）"
python3 -V
python3 - <<'PY' 2>/dev/null || echo "系统 python 没有 torch（正常，venv 里再装）"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device", p.name, f"sm{p.major}{p.minor}", "total", p.total_memory // 2**20, "MiB")
PY

echo
echo "== 磁盘 / 内存 =="
df -h "$HOME" | tail -1
free -g | awk 'NR<=2'

echo
echo "== docker 与 GPU 透传 =="
if command -v docker >/dev/null 2>&1; then
  docker --version
  if docker info 2>/dev/null | grep -qi nvidia; then
    echo "docker: 检测到 nvidia runtime（GPU 透传可能可用，仍需 --gpus all 实测）"
  else
    echo "docker: 未见 nvidia runtime —— D11 才正式搞容器，今天别在这上面耗"
  fi
else
  echo "docker: 未安装（今天不需要）"
fi

echo
echo "== 已有模型与 venv =="
ls -lh "$HOME/models" 2>/dev/null || echo "~/models 不存在"
ls -d "$HOME"/venv-vllm 2>/dev/null || echo "venv-vllm 未建（跑 scripts/d10_pip.sh）"
echo
echo "判定顺序：先 python3 bench/vram_probe.py，再 mem_budget.py，最后才装 vLLM。"
