#!/usr/bin/env bash
# D1 用：在 WSL 里跑，自动采集环境信息并生成 docs/env.md
# 用法: ./scripts/collect_env.sh [模型路径]
set -uo pipefail
MODEL_PATH=${1:-$HOME/models/qwen2.5-3b-instruct-q4_k_m.gguf}
LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
OUT=docs/env.md
mkdir -p docs

have() { command -v "$1" >/dev/null 2>&1 && echo "已装" || echo "未装"; }

{
  echo "# 实验环境记录（脚本采集 $(date '+%Y-%m-%d %H:%M')）"
  echo
  echo "> 本文件由 \`scripts/collect_env.sh\` 生成，Windows 侧字段需手工补（WSL 拿不到 Windows 驱动版本）。"
  echo
  echo "## GPU"
  echo
  echo '```'
  nvidia-smi 2>/dev/null || echo "nvidia-smi 不可用：WSL GPU 直通未生效，先更新 Windows 侧 NVIDIA 驱动"
  echo '```'
  echo
  echo "## 系统"
  echo
  echo '```'
  echo "wsl:      ${WSL_DISTRO_NAME:-unknown} (kernel $(uname -r))"
  echo "cpu:      $(lscpu 2>/dev/null | grep -m1 'Model name' | sed 's/.*: *//') | $(nproc) threads"
  echo "mem:      $(free -h 2>/dev/null | awk '/^Mem:/{print $2}')"
  echo "disk /:   $(df -h / | awk 'NR==2{print $4" 可用"}')"
  echo '```'
  echo
  echo "## 工具链"
  echo
  echo '```'
  echo "g++:      $(g++ --version 2>/dev/null | head -1)"
  echo "cmake:    $(cmake --version 2>/dev/null | head -1)"
  echo "nvcc:     $(nvcc -V 2>/dev/null | grep -m1 'release') $(command -v nvcc >/dev/null || echo '(PATH 里没有，见 docs/env.md 修复段)')"
  echo "python3:  $(python3 --version 2>/dev/null)"
  echo "docker:   $(have docker)"
  echo '```'
  echo
  echo "## llama.cpp"
  echo
  echo '```'
  if [ -d "$LLAMA_DIR/.git" ]; then
    echo "path:     $LLAMA_DIR"
    echo "commit:   $(git -C "$LLAMA_DIR" log -1 --format='%h %ad' --date=short)"
    echo "bin:      $(ls "$LLAMA_DIR"/build/bin/llama-cli "$LLAMA_DIR"/build/bin/llama-server 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')"
  else
    echo "未找到 $LLAMA_DIR（先 git clone 并编译）"
  fi
  echo '```'
  echo
  echo "## 模型"
  echo
  echo '```'
  if [ -f "$MODEL_PATH" ]; then
    echo "file:     $(basename "$MODEL_PATH")"
    echo "bytes:    $(stat -c%s "$MODEL_PATH")"
    echo "sha256:   $(sha256sum "$MODEL_PATH" | cut -c1-16)…（完整值耗时较长，按需再算）"
  else
    echo "未找到模型：$MODEL_PATH"
  fi
  echo '```'
  echo
  echo "## 待手工补（Windows 侧）"
  echo
  echo "- [ ] GPU 型号与显存总量、compute capability"
  echo "- [ ] Windows 驱动版本、nvidia-smi 显示的 CUDA 版本"
  echo "- [ ] 编译参数里的 -DCMAKE_CUDA_ARCHITECTURES 实际值"
  echo "- [ ] 显存档位结论（对应 env.md 第 6 节表格）"
} > "$OUT"

echo "[done] 已生成 $OUT"
sed -n '1,40p' "$OUT"
