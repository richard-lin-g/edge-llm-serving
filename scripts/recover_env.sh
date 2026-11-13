#!/usr/bin/env bash
# ============================================================
# EdgeServe 环境重建（D4 恢复版） · 2026-11-12
#
# 前提：Ubuntu 已装好（WSL2 或实体机/双系统均可）
# 用法：
#   bash recover_env.sh              # 全流程
#   bash recover_env.sh --dry-run    # 只探测，不动手
#   bash recover_env.sh --stage 3    # 只跑到第 3 步（可反复重入）
#
# 幂等：每一步先检测「是否已完成」，完成就跳过，不会重装。
# 日志：~/recover-<时间>.log（全流程 tee 一份，出问题直接发我）
# ============================================================
set -uo pipefail

STAGE_ALL=1
DRY=0
START_STAGE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --stage) START_STAGE="${2:-0}"; shift ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
  shift
done

LOG="$HOME/recover-$(date +%Y%m%d-%H%M).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== 日志: $LOG ==="
echo

# ---------- 探测 ----------
IS_WSL=0
grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=1
[ -n "${WSL_DISTRO_NAME:-}" ] && IS_WSL=1

CODENAME=$(. /etc/os-release 2>/dev/null && echo "${VERSION_CODENAME:-jammy}")
ARCH_CC="${ARCH_CC:-75}"          # T600 = Turing = 7.5；实体机换卡时覆盖这里

echo "=== 0. 环境探测 ==="
echo "运行方式    : $([ $IS_WSL -eq 1 ] && echo "WSL2 (${WSL_DISTRO_NAME:-?})" || echo "实体机/双系统")"
echo "发行版      : $(. /etc/os-release && echo "$PRETTY_NAME") ($CODENAME)"
echo "内核        : $(uname -r)"
echo "CPU 线程    : $(nproc)"
echo "内存        : $(free -h | awk '/^Mem:/{print $2}')"
echo "磁盘 / 可用 : $(df -h / | awk 'NR==2{print $4}')"
echo "用户        : $(whoami)  ($HOME)"
echo "CUDA 架构号 : $ARCH_CC"
echo
echo "--- GPU ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version \
             --format=csv,noheader || echo "nvidia-smi 在但查询失败"
else
  echo "!! 没有 nvidia-smi"
  if [ $IS_WSL -eq 1 ]; then
    echo "   WSL2 里没有 nvidia-smi = GPU 直通没生效。"
    echo "   处理：Windows 侧更新 NVIDIA 驱动（不要装 Linux 驱动），然后 wsl --shutdown 重开。"
  else
    echo "   实体机：需要装 NVIDIA Linux 驱动（nvidia-driver-570 或更新），装完重启。"
  fi
  echo "   没有 GPU 也能继续（会编译成 CPU 版），但今天的指标会全变，等于退回 CPU 基线。"
fi
echo

# ---------- 1. 基础工具链 ----------
stage1() {
  echo "=== 1. 基础工具链 ==="
  local need=()
  for p in g++ cmake git curl pkg-config python3; do
    command -v "$p" >/dev/null 2>&1 || need+=("$p")
  done
  if [ ${#need[@]} -eq 0 ]; then
    echo "[skip] 已齐：$(g++ --version | head -1)"
  else
    echo "[run ] 缺 ${need[*]}，安装中（会要 sudo 密码，请在这一步输入）"
    [ $DRY -eq 1 ] && { echo "[dry ] sudo apt update && sudo apt install -y ..."; return; }
    sudo apt update -y && sudo apt install -y \
      build-essential cmake git curl libcurl4-openssl-dev pkg-config python3-pip python3-venv
  fi
  python3 -c "import requests" 2>/dev/null || { echo "[run ] pip 装 requests"; [ $DRY -eq 0 ] && pip3 install -q requests --break-system-packages 2>/dev/null || pip3 install -q requests; }
  echo
}

# ---------- 2. CUDA Toolkit ----------
stage2() {
  echo "=== 2. CUDA Toolkit 12.6 ==="
  if command -v nvcc >/dev/null 2>&1; then
    echo "[skip] nvcc 已在 PATH：$(nvcc -V | grep -m1 release)"
  elif ls /usr/local/cuda-*/bin/nvcc >/dev/null 2>&1; then
    echo "[fix ] nvcc 已装只是不在 PATH，补 PATH"
    NV=$(ls -d /usr/local/cuda-12* 2>/dev/null | head -1)
    [ $DRY -eq 0 ] && {
      grep -q "$NV/bin" ~/.profile 2>/dev/null || {
        echo "export PATH=$NV/bin:\$PATH" >> ~/.profile
        echo "export LD_LIBRARY_PATH=$NV/lib64:\$LD_LIBRARY_PATH" >> ~/.profile
      }
    }
    echo "      已写入 ~/.profile（新终端或 source ~/.profile 后生效）"
  else
    echo "[run ] 未装，从 NVIDIA 官方源装 cuda-toolkit-12-6（约 2 GB，10–25 分钟）"
    if [ $DRY -eq 1 ]; then
      echo "[dry ] 按 $([ $IS_WSL -eq 1 ] && echo wsl-ubuntu || echo ubuntu2204) 源安装"
      return
    fi
    local REPO
    if [ $IS_WSL -eq 1 ]; then
      REPO="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64"
    else
      REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
    fi
    cd /tmp || return 1
    wget -q "$REPO/cuda-keyring_1.1-1_all.deb" -O cuda-keyring.deb \
      || { echo "[fail] 下 cuda-keyring 失败，检查网络/代理"; return 1; }
    sudo dpkg -i cuda-keyring.deb
    sudo apt update -y
    sudo apt install -y cuda-toolkit-12-6 \
      || { echo "[fail] cuda-toolkit 安装失败（常见：磁盘不足 / 网络中断），重跑本步即可续装"; return 1; }
    NV=$(ls -d /usr/local/cuda-12* 2>/dev/null | head -1)
    grep -q "$NV/bin" ~/.profile 2>/dev/null || {
      echo "export PATH=$NV/bin:\$PATH" >> ~/.profile
      echo "export LD_LIBRARY_PATH=$NV/lib64:\$LD_LIBRARY_PATH" >> ~/.profile
    }
    export PATH="$NV/bin:$PATH"
    echo "[ok  ] $(nvcc -V | grep -m1 release)"
    echo "       注意：不要装 nvidia-cuda-toolkit（那是 Ubuntu 自带 11.x，会和 12.6 冲突）"
  fi
  echo
}

# ---------- 3. llama.cpp ----------
stage3() {
  echo "=== 3. llama.cpp（锁定 b12033 保证数据可比）==="
  local DIR="$HOME/llama.cpp"
  if [ -x "$DIR/build/bin/llama-server" ] && ldd "$DIR/build/bin/llama-cli" 2>/dev/null | grep -qi cuda; then
    echo "[skip] 已有 CUDA 版：$(git -C "$DIR" log -1 --format='%h' 2>/dev/null)"
    echo
    return
  fi
  if [ ! -d "$DIR/.git" ]; then
    echo "[run ] clone"
    [ $DRY -eq 1 ] && { echo "[dry ] git clone + checkout 785322d17 + cmake"; echo; return; }
    git clone --quiet https://github.com/ggml-org/llama.cpp "$DIR" || { echo "[fail] clone 失败"; return 1; }
  fi
  cd "$DIR" || return 1
  # 锁定 D1/D2 实测所用 commit，数字才可比
  if git cat-file -e 785322d17^{commit} 2>/dev/null; then
    git checkout --quiet 785322d17 && echo "[ok  ] 已切到 785322d17（与 D1/D2 同版本）"
  else
    echo "[warn] 找不到 785322d17，用当前 HEAD：$(git log -1 --format='%h %ad' --date=short)"
    echo "       这意味着 D1/D2 的 6 档数据需重测后才能引用"
  fi
  echo "[run ] 配置 + 编译（$(nproc) 线程，20–40 分钟）"
  [ $DRY -eq 1 ] && { echo "[dry ] cmake -B build -DGGML_CUDA=ON ..."; echo; return; }
  rm -rf build
  cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCH_CC" \
    || { echo "[fail] cmake 配置失败（多半是 nvcc 还不在 PATH：source ~/.profile 后重跑）"; return 1; }
  cmake --build build -j"$(nproc)" \
    || { echo "[fail] 编译失败，日志在上面；常见是内存不足（-j 降到 4 重试）"; return 1; }
  echo "[check] ldd："
  ldd build/bin/llama-cli | grep -Ei "cuda|cublas" || echo "!! 没链接到 CUDA —— 说明编成了 CPU-only，删 build 重配"
  echo
}

# ---------- 4. 模型 ----------
stage4() {
  echo "=== 4. 3B Q4_K_M 模型（2.1 GB）==="
  local DIR="$HOME/models"
  local F="$DIR/qwen2.5-3b-instruct-q4_k_m.gguf"
  local WANT=2104932768
  mkdir -p "$DIR"
  if [ -f "$F" ] && [ "$(stat -c%s "$F")" -eq "$WANT" ]; then
    echo "[skip] 已存在且大小正确：$WANT 字节"
    echo
    return
  fi
  echo "[run ] 从 hf-mirror 下载（断点续传，10–25 分钟）"
  echo "       同时可以去查 Windows 侧有没有旧的 gguf 可拷，省这一步："
  echo "       find /mnt/c/Users -iname '*qwen2.5-3b*.gguf' 2>/dev/null"
  [ $DRY -eq 1 ] && { echo "[dry ] curl -L -C - hf-mirror ..."; echo; return; }
  curl -L -C - --retry 3 --retry-delay 2 \
    "https://hf-mirror.com/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf" \
    -o "$F" || { echo "[fail] 下载中断，重跑本步会断点续传"; return 1; }
  local GOT; GOT=$(stat -c%s "$F")
  [ "$GOT" -eq "$WANT" ] || { echo "[fail] 大小不符：$GOT / $WANT，删掉重下"; return 1; }
  echo "[ok  ] $GOT 字节"
  echo
}

# ---------- 5. 冒烟 ----------
stage5() {
  echo "=== 5. 冒烟：出第一个 token ==="
  local BIN="$HOME/llama.cpp/build/bin/llama-cli"
  local M="$HOME/models/qwen2.5-3b-instruct-q4_k_m.gguf"
  [ -x "$BIN" ] || { echo "[skip] 没有 llama-cli"; return; }
  [ $DRY -eq 1 ] && { echo "[dry ] llama-cli -ngl 36 -p 你好"; echo; return; }
  "$BIN" -m "$M" -ngl 36 -c 4096 -n 32 --no-warmup -p "你好，用一句话介绍你自己" </dev/null 2>&1 | tail -6
  echo
  echo "若上面出现 [ Prompt: x t/s | Generation: y t/s ]，且 Generation ≈ 38–40，"
  echo "说明重建成功、与 D1/D2 同档（39.33 ±10%）。低于 30 就先查显存占用。"
  echo
}

# ---------- 执行 ----------
run_from() {
  local from=$1
  [ "$from" -le 1 ] && stage1
  [ "$from" -le 2 ] && stage2
  [ "$from" -le 3 ] && stage3
  [ "$from" -le 4 ] && stage4
  [ "$from" -le 5 ] && stage5
}

run_from "$START_STAGE"

echo "============================================================"
echo "重建完毕。接下来（D4 本体）："
echo "  cd ~/edge-llm-serving"
echo "  # 1) 起服务（新终端，保持前台）"
echo "  ~/llama.cpp/build/bin/llama-server -m ~/models/qwen2.5-3b-instruct-q4_k_m.gguf \\
echo "      -ngl 36 -c 4096 --host 0.0.0.0 --port 8080"
echo "  # 2) 验 SSE 逐字（另一个终端）"
echo "  ./scripts/d4_sse_check.sh"
echo "  python3 bench/metrics.py --tag a-3b-ngl36 --base-url http://127.0.0.1:8080 \\"
echo "      --prompts bench/prompts.jsonl --repeat 3 --out results/d3_metrics.csv"
echo "============================================================"
echo "日志：$LOG"
