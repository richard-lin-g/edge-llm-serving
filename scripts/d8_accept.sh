#!/bin/bash
# D8 验收（12 项）
#   bash scripts/d8_accept.sh
# 全绿才 commit + push。黄灯不阻塞，但要在 docs/log/d8.md 里写一句为什么。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PASS=0; FAIL=0; WARN=0
ok(){ PASS=$((PASS+1)); printf '  \033[32m[PASS]\033[0m %s\n' "$1"; }
bad(){ FAIL=$((FAIL+1)); printf '  \033[31m[FAIL]\033[0m %s\n' "$1"; }
warn(){ WARN=$((WARN+1)); printf '  \033[33m[WARN]\033[0m %s\n' "$1"; }
chk(){ [ "$1" = "0" ] && ok "$2" || bad "$2"; }

echo "== D8 验收 =="

MODEL="$HOME/models/qwen2.5-3b-instruct-q4_k_m.gguf"
SZ=$(stat -c %s "$MODEL" 2>/dev/null || echo 0)
GIB=$(awk -v s="$SZ" 'BEGIN{printf "%.2f", s/1073741824}')
[ "$SZ" -gt 1900000000 ] && ok "模型在位（${GIB} GiB）" || bad "模型缺失或大小不对：$MODEL"

# 1. 模型参数
if [ -s results/d8_kv_model.json ]; then
  L=$(python3 -c "import json;d=json.load(open('results/d8_kv_model.json'));print(d.get('layers',0))")
  PT=$(python3 -c "import json;d=json.load(open('results/d8_kv_model.json'));print(d.get('kv_bytes_per_token',0))")
  [ "${L:-0}" -gt 0 ] && [ "${PT:-0}" -gt 0 ] && ok "d8_kv_model.json 有 layers=$L、每 token ${PT} B" || bad "d8_kv_model.json 字段不全"
else
  bad "缺 results/d8_kv_model.json（先跑 d8_gguf_meta.py）"
fi

# 2. 三档 ctx 都有行
ROWS=$(tail -n +2 results/d8_kv_mem.csv 2>/dev/null | grep -c . )
[ "$ROWS" -ge 3 ] && ok "d8_kv_mem.csv 有 $ROWS 行（≥3 档 ctx）" || bad "d8_kv_mem.csv 只有 $ROWS 行"

# 3. 差值比对全 PASS
if [ -s results/d8_kv_mem_checked.csv ]; then
  NF=$(grep -c "FAIL" results/d8_kv_mem_checked.csv || true)
  NP=$(grep -c "PASS" results/d8_kv_mem_checked.csv || true)
  [ "${NF:-0}" -eq 0 ] && ok "公式 vs 实测：$NP 行全部在容差内" || bad "有 $NF 行偏差超容差，先查 -ngl 与桌面占用"
else
  bad "缺 d8_kv_mem_checked.csv"
fi

# 4. 日志里的 KV 大小至少抓到一行（可比对的第二证据）
if grep -qE "ok" results/d8_kv_mem.csv 2>/dev/null; then
  KL=$(awk -F, '$5!="na" && $5!="" {c++} END{print c+0}' results/d8_kv_mem.csv)
  [ "$KL" -ge 1 ] && ok "有 $KL 行抓到了日志里的 KV 大小（第二证据）" \
                  || warn "没抓到 llama.cpp 的 KV 打印行，只靠显存差值判定（可在报告里注明）"
fi

# 5. 深度曲线
DR=$(tail -n +2 results/d8_depth.csv 2>/dev/null | grep -c . )
[ "$DR" -ge 4 ] && ok "d8_depth.csv 有 $DR 个深度档" || bad "d8_depth.csv 只有 $DR 档（要 ≥4：512/2K/4K/8K）"
if [ "$DR" -ge 4 ]; then
  python3 - <<'PY'
import csv
rows=list(csv.DictReader(open("results/d8_depth.csv")))
tg=[float(r["tg_tps"]) for r in rows if r["tg_tps"] not in ("","na")]
if len(tg)>=2:
    drop=(max(tg)-min(tg))/max(tg)*100
    print(f"  · tg 从最深到最浅落差 {drop:.1f}%（最浅 {max(tg):.1f} → 最深 {min(tg):.1f} t/s）")
    if drop>40: print("  · 超过 40%：先查 -ngl 是否满载 / swap，再决定是否写成“KV 带宽瓶颈”")
PY
fi

# 6. 调优表
TR=$(tail -n +2 results/d8_tune.csv 2>/dev/null | grep -c . )
[ "$TR" -ge 7 ] && ok "d8_tune.csv 有 $TR 个组合（≥7）" || bad "d8_tune.csv 只有 $TR 个组合（6 组 batch + ≥1 组线程）"

# 7. 报告章节
if [ -s docs/kv-cache.md ]; then
  SEC=$(grep -c "^## " docs/kv-cache.md)
  [ "$SEC" -ge 6 ] && ok "docs/kv-cache.md 有 $SEC 节（≥6）" || bad "docs/kv-cache.md 只有 $SEC 节"
  if grep -qE "TODO|待填|待测|XXX|【填】" docs/kv-cache.md; then bad "docs/kv-cache.md 还有占位符未填"
  else ok "docs/kv-cache.md 无占位符"; fi
else
  bad "缺 docs/kv-cache.md"
fi

# 8. 日志
if [ -s docs/log/d8.md ]; then
  grep -qE "TODO|待填" docs/log/d8.md && bad "docs/log/d8.md 还有 TODO" || ok "docs/log/d8.md 已收尾"
else
  bad "缺 docs/log/d8.md"
fi

# 9. 提交数
CN=$(git rev-list --count HEAD 2>/dev/null || echo 0); CN=${CN:-0}
[ "$CN" -ge 7 ] && ok "仓库提交数 $CN（≥7）" || bad "提交数只有 $CN，今天的还没提交？"

# 10. 可选 KV 量化
[ -s results/d8_kvq.csv ] && ok "附加：KV 量化对照已采（$(tail -n +2 results/d8_kvq.csv | grep -c .) 行）" \
                          || warn "没做 KV 量化对照（可选，时间够再补）"

echo
echo "结果：$PASS 绿 / $WARN 黄 / $FAIL 红"
[ "$FAIL" -eq 0 ] && echo "可以提交：git add -A && git commit -m \"exp(kv): ctx 2k-8k vram vs formula, depth curve, batch/thread tuning\"" \
                  || echo "先修红的，别提交半成品。"
exit "$FAIL"
