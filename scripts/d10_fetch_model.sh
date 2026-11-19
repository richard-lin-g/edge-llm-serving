#!/usr/bin/env bash
# scripts/d10_fetch_model.sh — 下 AWQ W4 权重（vLLM 要 HF/safetensors 格式，不吃 GGUF）
# 为什么是 AWQ：fp16 的 6.18 GB 在 4 GB 卡上连权重都放不下，只有 4bit 路线有实验价值。
# 用法： bash scripts/d10_fetch_model.sh            # 默认 Qwen2.5-3B-Instruct-AWQ → ~/models/qwen25-3b-awq
#      MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ bash scripts/d10_fetch_model.sh
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}
DEST=${DEST:-$HOME/models/qwen25-3b-awq}

mkdir -p "$(dirname "$DEST")"

if [ -f "$DEST/config.json" ] && ls "$DEST"/*.safetensors >/dev/null 2>&1; then
  echo "已存在：$DEST"
  du -sh "$DEST"
  exit 0
fi

echo "== 下载 $MODEL → $DEST =="
# 路线 A：魔搭（国内最稳）
if python3 -c "import modelscope" >/dev/null 2>&1; then
  python3 - <<PY
from modelscope import snapshot_download
snapshot_download("$MODEL", local_dir="$DEST")
PY
else
  echo "未装 modelscope，先 pip install modelscope（或走路线 B）"
fi

# 路线 B：HF 镜像（魔搭失败时手动跑）
if [ ! -f "$DEST/config.json" ]; then
  echo "改用 hf-mirror 直下……"
  pip install -q "huggingface_hub[cli]"
  HF_ENDPOINT=https://hf-mirror.com hf download "$MODEL" --local-dir "$DEST"
fi

echo
echo "== 结果 =="
du -sh "$DEST"
ls -lh "$DEST" | head -12
echo
echo "把上面的实际体积记进报告：这就是「W4 权重」这一行的真值（预期约 2.0–2.2 GB）。"
