#!/usr/bin/env bash
# D5 验收：证明这个客户端真的在「逐 token 流式」消费，而不是等整包
# 用法: ./scripts/d5_accept.sh [base_url] [client_bin]
#   默认 http://127.0.0.1:8080  与 ./build/edge-client（在 client-cpp 目录下执行）
set -uo pipefail

BASE=${1:-http://127.0.0.1:8080}
BIN=${2:-./build/edge-client}
HOSTPORT=${BASE#*://}
HOST=${HOSTPORT%%:*}
PORT=${HOSTPORT##*:}
OUT=logs/d5
mkdir -p "$OUT"

PROMPT="用三句话说明 KV Cache 为什么能加速解码。"
PASS=0
FAIL=0
ok()   { echo "  [PASS] $*"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
warn() { echo "  [WARN] $*"; }

echo "== 0) 服务探活 =="
if curl -sf "$BASE/health" >/dev/null; then ok "health ok"; else
  bad "服务没起来：$BASE/health 不通。先看服务日志第一行报错"; echo; echo "结论：未通过"; exit 1
fi

echo
echo "== 1) 编译产物存在 =="
[ -x "$BIN" ] && ok "找到可执行: $BIN" || bad "找不到 $BIN，先 cmake --build build -j"

echo
echo "== 2) 流式单轮 =="
STAMP=$(date +%Y%m%d-%H%M%S)
"$BIN" -H "$HOST" -p "$PORT" -m "$PROMPT" -n 96 > "$OUT/stream-$STAMP.txt" 2> "$OUT/stream-$STAMP.err"
RC=$?
if [ $RC -eq 0 ]; then ok "退出码 0"; else bad "退出码 $RC（看 $OUT/stream-$STAMP.err）"; fi

TXT=$(tr -d '\n' < "$OUT/stream-$STAMP.txt")
[ -n "$TXT" ] && ok "有正文输出（${#TXT} 字符）" || bad "正文为空"

STATS=$(grep '\[stats\]' "$OUT/stream-$STAMP.err" | tail -1)
echo "  $STATS"

TTFT=$(echo "$STATS" | grep -o 'ttft=[0-9.]*' | cut -d= -f2)
TOTAL=$(echo "$STATS" | grep -o 'total=[0-9.]*' | cut -d= -f2)
PIECES=$(echo "$STATS" | grep -o 'pieces=[0-9]*' | cut -d= -f2)
DECODE=$(echo "$STATS" | grep -o 'decode=[0-9.]*' | cut -d= -f2)
TOKS=$(echo "$STATS" | grep -o 'tokens=[0-9-]*' | cut -d= -f2)

if [ "${PIECES:-0}" -gt 1 ]; then ok "收到 $PIECES 个非空 delta（真流式）"; else
  bad "只收到 1 片：服务端把整包发回来了"; fi

# 核心判据：首字延迟必须明显小于总耗时，否则等于「攒完才给」
if [ -n "${TTFT:-}" ] && [ -n "${TOTAL:-}" ] && [ -n "${PIECES:-}" ] && [ "$PIECES" -gt 1 ]; then
  RATIO=$(python3 -c "print(f'{$TTFT / $TOTAL:.3f}')" 2>/dev/null || echo 1)
  echo "  TTFT/总耗时 = $RATIO"
  python3 -c "import sys; sys.exit(0 if $TTFT < 0.6 * $TOTAL else 1)" \
    && ok "TTFT($TTFT ms) < 60% 总耗时($TOTAL ms)：流式成立" \
    || bad "TTFT 接近总耗时，疑似被缓冲成整包（查反代 / stream 参数）"
else
  bad "统计行缺字段，无法判定"
fi

if [ -n "${DECODE:-}" ]; then ok "解码速度 ${DECODE} tok/s"; else warn "未取到 decode，检查服务端是否返回 usage/timings"; fi
if [ "${TOKS:-0}" -gt 0 ]; then ok "tokens=${TOKS}"; else warn "tokens 字段异常"; fi

echo
echo "== 3) 非流式对照（同 prompt、temperature=0）=="
"$BIN" -H "$HOST" -p "$PORT" -m "$PROMPT" -n 96 --no-stream > "$OUT/nostream-$STAMP.txt" 2> "$OUT/nostream-$STAMP.err"
if [ $? -eq 0 ]; then ok "--no-stream 正常返回"; else bad "--no-stream 失败"; fi
if diff -q "$OUT/stream-$STAMP.txt" "$OUT/nostream-$STAMP.txt" >/dev/null 2>&1; then
  ok "两条路径正文一致（拼接逻辑没错）"
else
  warn "两条路径正文有差异（贪心采样下偶发；若差异很大说明 delta 拼接有 bug）"
  diff "$OUT/stream-$STAMP.txt" "$OUT/nostream-$STAMP.txt" | head -5
fi

echo
echo "== 4) 断线重连 =="
"$BIN" -H "$HOST" -p 1 -m "x" > /dev/null 2> "$OUT/reconnect-$STAMP.err"
grep -q '\[retry\]' "$OUT/reconnect-$STAMP.err" && ok "连接失败会重试一次" || bad "没有走到重连逻辑"
grep -q '\[error\]' "$OUT/reconnect-$STAMP.err" && ok "重试仍失败时正确报错退出" || bad "重试后既没成功也没报错"

echo
echo "== 结论 =="
echo "PASS=$PASS  FAIL=$FAIL"
echo "原始输出已存: $OUT/（stream / nostream / 两份 .err）"
if [ "$FAIL" -eq 0 ]; then
  echo "[PASS] D5 验收通过 —— 客户端可编译、可对话、真流式"
  exit 0
else
  echo "[FAIL] 有 $FAIL 项未通过，按上面提示逐条修"
  exit 1
fi
