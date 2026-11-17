#!/bin/bash
# D8 · -b / -ub / -t 调优
#
# 要点：-b/-ub 影响的是 pp（prompt processing）的批处理效率，对 tg 基本无影响；
#       -t 在 GPU 满载（-ngl 36）时只影响少量 CPU 侧工作，收益有限。
#       本机 llama-bench（785322d）没有 -c，深度用 -d。
#
#   bash scripts/d8_tune.sh
#   ONLY="b512" bash scripts/d8_tune.sh      # 只跑名字含 b512 的档
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BIN="${BIN:-$HOME/llama.cpp/build/bin}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL="${MODEL:-$MODEL_DIR/qwen2.5-3b-instruct-q4_k_m.gguf}"
NGL="${NGL:-36}"
NPRED="${NPRED:-64}"
ONLY="${ONLY:-}"
OUT="results/d8_tune.csv"

[ -x "$BIN/llama-bench" ] || { echo "! 缺 $BIN/llama-bench"; exit 1; }
[ -s "$MODEL" ] || { echo "! 缺模型 $MODEL"; exit 1; }

mkdir -p results
echo "name,b,ub,t,ngl,pp_tps,pp_std,tg_tps,tg_std,vram_peak_mib,notes" > "$OUT"

vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]'; }

run_one() {   # name b ub t
  local NAME=$1 B=$2 UB=$3 TH=$4
  if [ -n "$ONLY" ] && [[ "$NAME" != *"$ONLY"* ]]; then return 0; fi
  local LOG="results/d8_tune_${NAME}.log"
  echo "== $NAME  -b $B -ub $UB -t $TH =="

  BASE=$(vram)
  "$BIN/llama-bench" -m "$MODEL" -p 512 -n "$NPRED" -r 3 -d 0 \
      -b "$B" -ub "$UB" -t "$TH" -ngl "$NGL" -o md > "$LOG" 2>&1 &
  PID=$!
  PEAK=$BASE
  while kill -0 "$PID" 2>/dev/null; do
    V=$(vram); [ -n "${V:-}" ] && [ "$V" -gt "$PEAK" ] 2>/dev/null && PEAK=$V
    sleep 0.4
  done
  wait "$PID"; RC=$?

  if [ $RC -ne 0 ]; then
    if grep -qiE "out of memory|OOM|failed to allocate" "$LOG"; then
      NOTE="OOM"
    else
      NOTE="rc=$RC"
    fi
    echo "  ! 失败（$NOTE）"; tail -3 "$LOG" | sed 's/^/    /'
    echo "$NAME,$B,$UB,$TH,$NGL,na,na,na,na,$PEAK,$NOTE" >> "$OUT"
    return 0
  fi

  PP=$(grep -oE "pp512[^|]*\| *[0-9.]+" "$LOG" | grep -oE "[0-9.]+$")
  PPS=$(grep -oE "pp512[^|]*\| *[0-9.]+ ± [0-9.]+" "$LOG" | grep -oE "± [0-9.]+" | grep -oE "[0-9.]+")
  TG=$(grep -oE "tg${NPRED}[^|]*\| *[0-9.]+" "$LOG" | grep -oE "[0-9.]+$")
  TGS=$(grep -oE "tg${NPRED}[^|]*\| *[0-9.]+ ± [0-9.]+" "$LOG" | grep -oE "± [0-9.]+" | grep -oE "[0-9.]+")

  echo "  pp ${PP:-na} t/s   tg ${TG:-na} t/s   显存峰值 ${PEAK} MiB"
  echo "$NAME,$B,$UB,$TH,$NGL,${PP:-na},${PPS:-na},${TG:-na},${TGS:-na},$PEAK,ok" >> "$OUT"
  sleep 2
}

# 基线：默认值
run_one "baseline"     2048 512  8
# 只动 -b：逻辑批更小，看 pp 是否掉
run_one "b1024_ub512"  1024 512  8
run_one "b512_ub512"   512  512  8
# 只动 -ub：物理批，直接影响 GPU 计算粒度
run_one "b2048_ub256"  2048 256  8
run_one "b2048_ub128"  2048 128  8
run_one "b2048_ub1024" 2048 1024 8
# 大 ub 组合
run_one "b4096_ub1024" 4096 1024 8
# 只动 -t
run_one "t6"           2048 512  6
run_one "t12"          2048 512  12
run_one "t16"          2048 512  16

echo
echo "== 调优结果（按 pp 降序）=="
python3 - "$OUT" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["pp_tps"] not in ("na", "")]
rows.sort(key=lambda r: -float(r["pp_tps"]))
print(f"{'name':<16}{'b':>6}{'ub':>6}{'t':>4}{'pp t/s':>9}{'tg t/s':>9}{'VRAM':>8}")
for r in rows:
    print(f"{r['name']:<16}{r['b']:>6}{r['ub']:>6}{r['t']:>4}{float(r['pp_tps']):>9.2f}{float(r['tg_tps']):>9.2f}{r['vram_peak_mib']:>8}")
bad = [r for r in csv.DictReader(open(sys.argv[1])) if r["pp_tps"] in ("na", "")]
if bad:
    print("\n失败档：")
    for r in bad:
        print(f"  {r['name']:<16} b={r['b']} ub={r['ub']} t={r['t']}  {r['notes']}")
PY
echo
echo "结论怎么读：pp 最高且 tg 不掉、显存不超 3.4 GB 的那一组，就是 4 GB 卡上的推荐值。"
