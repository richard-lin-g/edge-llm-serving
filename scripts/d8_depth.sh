#!/bin/bash
# D8 · 深度-速度曲线：同一组参数下，KV 里"已缓存深度"从 512 拉到 8192
#
# 本机 llama-bench（build 785322d）没有 -c，深度用 -d 指定：
#   -d 2048 表示先垫 2048 token 的 KV，再测 pp512 与 tg64 —— 正是要的"该深度下的速度"。
#
#   bash scripts/d8_depth.sh
#   DEPTHS="512 2048 4096" bash scripts/d8_depth.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BIN="${BIN:-$HOME/llama.cpp/build/bin}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL="${MODEL:-$MODEL_DIR/qwen2.5-3b-instruct-q4_k_m.gguf}"
NGL="${NGL:-36}"
T="${T:-8}"
NPRED="${NPRED:-64}"
DEPTHS="${DEPTHS:-0 2048 4096 8192}"
OUT="results/d8_depth.csv"

[ -x "$BIN/llama-bench" ] || { echo "! 缺 $BIN/llama-bench"; exit 1; }
[ -s "$MODEL" ] || { echo "! 缺模型 $MODEL"; exit 1; }

mkdir -p results
COMMIT="${COMMIT:-$(git -C "$HOME/llama.cpp" log -1 --format=%h 2>/dev/null || echo unknown)}"
rm -f "$OUT"

for D in $DEPTHS; do
  LOG="results/d8_depth_${D}.log"
  echo "== depth=$D  llama-bench -p 512 -n $NPRED -d $D -ngl $NGL =="
  "$BIN/llama-bench" -m "$MODEL" -p 512 -n "$NPRED" -r 3 -d "$D" \
      -ngl "$NGL" -t "$T" -o md > "$LOG" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "  ! 失败（rc=$RC），末尾："; tail -4 "$LOG" | sed 's/^/    /'; continue
  fi
  python3 bench/d8_parse_bench.py "$LOG" --depth "$D" --ngl "$NGL" \
      --commit "$COMMIT" --out "$OUT"
done

echo
echo "== 曲线 =="
if [ -s "$OUT" ]; then
  column -s, -t < "$OUT" 2>/dev/null || cat "$OUT"
  python3 - "$OUT" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if len(rows) >= 2:
    pp = [(int(r["depth"]), float(r["pp_tps"])) for r in rows]
    tg = [(int(r["depth"]), float(r["tg_tps"])) for r in rows]
    base_pp, base_tg = pp[0][1], tg[0][1]
    last_pp, last_tg = pp[-1][1], tg[-1][1]
    print()
    print(f"  pp: d{pp[0][0]} {base_pp:.2f} → d{pp[-1][0]} {last_pp:.2f} t/s，落差 {(1-last_pp/base_pp)*100:.1f}%")
    print(f"  tg: d{tg[0][0]} {base_tg:.2f} → d{tg[-1][0]} {last_tg:.2f} t/s，落差 {(1-last_tg/base_tg)*100:.1f}%")
PY
else
  echo "  ! 没有结果，全部档位都失败了"
fi
echo
echo "读法：tg 从 d0 拉到 d8192 掉了几个百分点？"
echo "   掉 <15% → 结论是“这一档规模下 KV 读量不是瓶颈，权重读量才是”，这是真结论，别硬凑戏剧性。"
echo "   掉 >30% → 先查 -ngl 是否满载、是否触发 swap，再决定要不要写进“KV 带宽瓶颈”。"
