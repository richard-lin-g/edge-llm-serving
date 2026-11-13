#!/usr/bin/env bash
# 一键起 llama-server。用法:
#   ./scripts/run_server.sh cpu             # apt 版，CPU 后端（D4 协议验证用这个）
#   ./scripts/run_server.sh gpu [ngl]       # ~/edge-llm-serving/bin 里的 CUDA 版（需驱动 >= 580）
# 可选环境变量: MODEL / PORT / CTX
#   MODEL 默认 ~/models/qwen2.5-3b-instruct-q4_k_m.gguf
#   PORT  默认 8080，CTX 默认 4096
set -uo pipefail
MODE=${1:-cpu}
NGL=${2:-99}
MODEL=${MODEL:-$HOME/models/qwen2.5-3b-instruct-q4_k_m.gguf}
PORT=${PORT:-8080}
CTX=${CTX:-4096}
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

case "$MODE" in
  cpu)
    BIN=/usr/bin/llama-server
    EXTRA="-ngl 0"
    ;;
  gpu)
    BIN=$PWD/bin/llama-server
    export LD_LIBRARY_PATH=$PWD/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    EXTRA="-ngl $NGL"
    ;;
  *)
    echo "用法: $0 {cpu|gpu} [ngl]"; exit 2 ;;
esac

[ -x "$BIN" ] || { echo "[FAIL] 找不到可执行文件: $BIN"; exit 1; }
[ -f "$MODEL" ] || { echo "[FAIL] 找不到模型: $MODEL"; exit 1; }

SZ=$(stat -c%s "$MODEL")
if [ "$SZ" != "2104932768" ]; then
  echo "[WARN] 模型大小 $SZ != 2104932768，可能下载不完整，先校验再起服务"
fi

echo "模式=$MODE  ngl=$EXTRA  port=$PORT  ctx=$CTX"
echo "模型=$MODEL ($SZ B)"
echo "日志=logs/serve-$MODE.log"
echo "提示: GPU 模式若报 'CUDA driver version is insufficient'，说明驱动低于 580，见 docs/env.md 3.2.1"
echo

"$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" -c "$CTX" $EXTRA --seed 42 2>&1 \
  | tee "logs/serve-$MODE.log"
