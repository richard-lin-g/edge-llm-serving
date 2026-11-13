#!/usr/bin/env bash
# D4 验收：证明 OpenAI 兼容接口真的在「逐字流式」返回
# 用法: ./scripts/d4_sse_check.sh [base_url]  默认 http://127.0.0.1:8080
#
# 计时口径说明：用 /proc/uptime 的单调钟取毫秒，不用 `date +%s%3N`。
# 原因：WSL2 的系统时钟跟随宿主机休眠/唤醒会跳变，实测出现过负数和
# +621289682ms 这种荒谬值；单调钟不受影响。精确 TTFT/TPOT 仍以
# bench/metrics.py 的结果为准，这里只看「分片是否逐个到达」。
set -uo pipefail
BASE=${1:-http://127.0.0.1:8080}
OUT=logs
mkdir -p "$OUT"

now_ms() { awk '{printf "%d", $1*1000}' /proc/uptime; }

echo "== 1) 探活 /health =="
if ! curl -sf "$BASE/health" | grep -q '"ok"'; then
  echo "[FAIL] 服务没起来或 health 不是 ok。先看 logs/serve-*.log 第一行报错"
  exit 1
fi
echo "[OK] health ok"

echo
echo "== 2) /v1/models 是否暴露模型 =="
curl -s "$BASE/v1/models" | head -c 400; echo

echo
echo "== 3) 流式请求（-N 关闭缓冲，观察逐字） =="
STAMP=$(date +%Y%m%d-%H%M%S)
RAW="$OUT/d4_sse-$STAMP.sse"
START=$(now_ms)

curl -sN "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [{"role":"user","content":"用三句话说明 KV Cache 为什么能加速解码。"}],
    "max_tokens": 96,
    "temperature": 0,
    "stream": true
  }' > "$RAW" &
CURL_PID=$!

# 逐个分片打印到达时刻，肉眼可见「不是一个整包回来」
FIRST=""
NCHUNK=0
while kill -0 "$CURL_PID" 2>/dev/null; do
  N=$(grep -c '^data: ' "$RAW" 2>/dev/null | head -1); [ -z "$N" ] && N=0
  if [ "$N" -gt "$NCHUNK" ]; then
    [ -z "$FIRST" ] && FIRST=$(now_ms)
    NCHUNK=$N
    printf '  chunk %-3s  +%sms\n' "$N" "$(( $(now_ms) - START ))"
  fi
  sleep 0.05
done
wait "$CURL_PID" 2>/dev/null

# 兜底：轮询可能漏计，以最终文件为准
FIN=$(grep -c '^data: ' "$RAW" 2>/dev/null | head -1); [ -z "$FIN" ] && FIN=0
[ "$FIN" -gt "$NCHUNK" ] && NCHUNK=$FIN

END=$(now_ms)
TOTAL=$(( END - START ))
[ -z "$FIRST" ] && TTFT="n/a" || TTFT=$(( FIRST - START ))

echo
echo "== 4) 拼回正文（应是一段连贯中文，不是空/乱码） =="
python3 - "$RAW" <<'PY'
import json, sys
raw = open(sys.argv[1], encoding='utf-8', errors='replace').read()
txt, usage, timings = [], None, None
for line in raw.splitlines():
    if not line.startswith('data: '):
        continue
    body = line[6:].strip()
    if body == '[DONE]':
        continue
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        continue
    usage = obj.get('usage') or usage
    timings = obj.get('timings') or timings
    for ch in obj.get('choices', []):
        piece = (ch.get('delta') or {}).get('content')
        if piece:
            txt.append(piece)
print(''.join(txt))
print('\n--- usage ---');   print(json.dumps(usage, ensure_ascii=False))
print('--- timings ---');   print(json.dumps(timings, ensure_ascii=False))
PY

echo
echo "== 结论 =="
echo "chunk 数：$NCHUNK（>1 且时间递增 = 真流式；=1 = 被缓冲成整包）"
echo "TTFT：${TTFT} ms（粗测，精确值用 bench/metrics.py --tag ... 取）"
echo "总耗时：${TOTAL} ms"
echo "原始分片已存：$RAW"
if [ "$NCHUNK" -gt 1 ]; then
  echo "[PASS] SSE 逐字返回正常 —— D4 验收项通过"
else
  echo "[FAIL] 只收到 1 个分片：检查是否有反代/缓冲，或 stream 参数没传进"
  exit 1
fi
