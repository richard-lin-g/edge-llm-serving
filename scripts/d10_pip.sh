#!/usr/bin/env bash
# scripts/d10_pip.sh — 建 venv 并装 vLLM（默认 0.8.5.post1：cu126 轮子、门槛仍是 7.5、体积比新版本小）
#
# 为什么先走 pip 而不是 docker：
#   1) D11 才正式搞容器，今天的 docker GPU 透传还没验证过，不该把唯一 4 小时赌在它上面；
#   2) pip 路径能直接复用 vram_probe.py 的 torch，一步拿到「可用显存」这个关键数。
#
# 驱动不够新就别硬装高版本：cu126 轮子要 560+ 的驱动；更老的驱动换 cu124（VLLM_VER=0.6.6）。
#
# 用法： bash scripts/d10_pip.sh
#      VLLM_VER=0.6.6 bash scripts/d10_pip.sh          # 驱动较老时
#      PYPI=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/d10_pip.sh
set -euo pipefail

VLLM_VER=${VLLM_VER:-0.8.5.post1}
PYPI=${PYPI:-https://pypi.tuna.tsinghua.edu.cn/simple}
VENV=${VENV:-$HOME/venv-vllm}

echo "== 驱动 =="
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "driver $DRV"
MAJOR=${DRV%%.*}
if [ "$MAJOR" -lt 560 ] 2>/dev/null; then
  echo "警告：驱动 < 560，cu126 轮子可能初始化失败；建议 VLLM_VER=0.6.6（cu124）。"
fi

echo
echo "== venv =="
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -V

echo
echo "== 装包（约 2.5–3.5 GB 下载，10–20 分钟）=="
pip install -U pip wheel -i "$PYPI" -q
pip install "vllm==$VLLM_VER" -i "$PYPI"
python - <<'PY'
import torch, vllm
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
print("vllm", vllm.__version__)
PY

echo
echo "== 下一步 =="
echo "source $VENV/bin/activate && python3 bench/vram_probe.py"
