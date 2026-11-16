#!/usr/bin/env bash
# D7 · 补齐三档量化：Q4_K_M（D2 已下）+ Q4_0 + Q8_0
# 用法: bash scripts/d7_get_models.sh
#   bash scripts/d7_get_models.sh --scan-windows   # 先看 /mnt/c 里有没有现成的
set -uo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
REPO="Qwen/Qwen2.5-3B-Instruct-GGUF"
SCAN=0
[ "${1:-}" = "--scan-windows" ] && SCAN=1
mkdir -p "$MODEL_DIR" results

# 文件 : 最小字节数（防止被截断还拿去跑）
need() {
  case "$1" in
    q4_0)   echo 1600000000 ;;
    q4_k_m) echo 1900000000 ;;
    q8_0)   echo 3000000000 ;;
  esac
}
file_of() { echo "qwen2.5-3b-instruct-$(echo "$1" | tr '_' '_').gguf"; }

if [ "$SCAN" = 1 ]; then
  echo "[扫描] /mnt/c/Users 下找 qwen2.5-3b*gguf（最多 60 秒）"
  timeout 60 find /mnt/c/Users -maxdepth 8 -iname 'qwen2.5-3b*gguf' 2>/dev/null | head -5
  echo "  有的话直接拷到 $MODEL_DIR ，省流量"
fi

echo "quant,file,size_bytes,size_gib" > results/d7_model_sizes.csv

for Q in q4_0 q4_k_m q8_0; do
  F="qwen2.5-3b-instruct-${Q}.gguf"
  T="$MODEL_DIR/$F"
  MIN=$(need "$Q")
  if [ -s "$T" ] && [ "$(stat -c %s "$T")" -ge "$MIN" ]; then
    SZ=$(stat -c %s "$T")
    echo "[已有] $F  $((SZ/1048576)) MiB"
  else
    echo "[下载] $F"
    curl -fL -C - --retry 5 --retry-delay 3 --connect-timeout 20 \
      -o "$T" "https://hf-mirror.com/$REPO/resolve/main/$F" || {
        echo "! $Q 下载失败。替代："
        echo "  1) 官方源: URL 换成 https://huggingface.co/$REPO/resolve/main/$F"
        echo "  2) 列出实际文件名: curl -s https://hf-mirror.com/api/models/$REPO | head"
        echo "  3) 明天再补；今天至少把已下载的档跑完，报告里标注“Q4_0 未取得”"
        continue
      }
    SZ=$(stat -c %s "$T")
  fi
  SZ=$(stat -c %s "$T" 2>/dev/null || echo 0)
  if [ "$SZ" -lt "$MIN" ]; then
    echo "! $F 只有 $((SZ/1048576)) MiB，小于阈值，删掉重下：rm $T"
    continue
  fi
  awk -v q="$Q" -v f="$F" -v b="$SZ" 'BEGIN{printf "%s,%s,%d,%.2f\n", q, f, b, b/1073741824}' >> results/d7_model_sizes.csv
done

echo
echo "== 三档核对 =="
cat results/d7_model_sizes.csv
echo "提示：Q8_0 约 3.2 GiB，4 GB 卡上要和 KV Cache 抢显存——跑不动就降 -ngl，脚本会自动退。"
