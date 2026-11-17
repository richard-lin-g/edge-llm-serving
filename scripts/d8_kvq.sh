#!/bin/bash
# D8 · KV 量化对照：f16 / q8_0 / q4_0 的显存与速度
#
# 原来的版本用 llama-cli 采集，但本机 llama-cli（build 785322d）在此环境无输出，
# 改为 llama-bench 的 -ctk / -ctv（cache-type-k / cache-type-v），一次跑出速度，
# 同时采样显存峰值。
#
#   bash scripts/d8_kvq.sh
#   CTX=4096 bash scripts/d8_kvq.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BIN="${BIN:-$HOME/llama.cpp/build/bin}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL="${MODEL:-$MODEL_DIR/qwen2.5-3b-instruct-q4_k_m.gguf}"
NGL="${NGL:-36}"
T="${T:-8}"
NPRED="${NPRED:-64}"
DEPTH="${DEPTH:-8192}"        # 用 8K 深度，KV 量化省显存的效果才看得出来
OUT="results/d8_kvq.csv"

[ -x "$BIN/llama-bench" ] || { echo "! 缺 $BIN/llama-bench"; exit 1; }
[ -s "$MODEL" ] || { echo "! 缺模型 $MODEL"; exit 1; }

mkdir -p results
echo "ctk,ctv,depth,ngl,pp_tps,tg_tps,vram_peak_mib,note" > "$OUT"

vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]'; }

run_q() {   # ctk ctv
  local CTK=$1 CTV=$2
  local NAME="${CTK}_${CTV}"
  local LOG="results/d8_kvq_${NAME}.log"
  echo "== ctk=$CTK ctv=$CTV  depth=$DEPTH =="

  BASE=$(vram)
  "$BIN/llama-bench" -m "$MODEL" -p 512 -n "$NPRED" -r 2 -d "$DEPTH" \
      -ctk "$CTK" -ctv "$CTV" -ngl "$NGL" -t "$T" -o md > "$LOG" 2>&1 &
  PID=$!
  PEAK=$BASE
  while kill -0 "$PID" 2>/dev/null; do
    V=$(vram); [ -n "${V:-}" ] && [ "$V" -gt "$PEAK" ] 2>/dev/null && PEAK=$V
    sleep 0.4
  done
  wait "$PID"; RC=$?

  if [ $RC -ne 0 ]; then
    local NOTE="rc=$RC"
    grep -qiE "out of memory|OOM|failed to allocate" "$LOG" && NOTE="OOM"
    grep -qiE "unsupported|not supported|invalid" "$LOG" && NOTE="unsupported"
    echo "  ! 失败（$NOTE）"; tail -3 "$LOG" | sed 's/^/    /'
    echo "$CTK,$CTV,$DEPTH,$NGL,na,na,$PEAK,$NOTE" >> "$OUT"
    sleep 2; return 0
  fi

  local PP TG
  PP=$(grep -oE "pp512[^|]*\| *[0-9.]+" "$LOG" | grep -oE "[0-9.]+$" | head -1)
  TG=$(grep -oE "tg${NPRED}[^|]*\| *[0-9.]+" "$LOG" | grep -oE "[0-9.]+$" | head -1)
  echo "  pp ${PP:-na}   tg ${TG:-na}   显存峰值 ${PEAK} MiB"
  echo "$CTK,$CTV,$DEPTH,$NGL,${PP:-na},${TG:-na},$PEAK,ok" >> "$OUT"
  sleep 2
}

# V 必须 f16：K 量化是主流做法；V 量化在部分后端不支持或明显掉精度
run_q f16  f16
run_q q8_0 f16
run_q q4_0 f16

echo
echo "== 汇总 =="
python3 - "$OUT" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(f"{'ctk':<6}{'ctv':<6}{'depth':>7}{'pp t/s':>9}{'tg t/s':>9}{'VRAM MiB':>10}")
for r in rows:
    pp = r['pp_tps'] if r['pp_tps'] == 'na' else f"{float(r['pp_tps']):.2f}"
    tg = r['tg_tps'] if r['tg_tps'] == 'na' else f"{float(r['tg_tps']):.2f}"
    print(f"{r['ctk']:<6}{r['ctv']:<6}{r['depth']:>7}{pp:>9}{tg:>9}{r['vram_peak_mib']:>10}")
ok = [r for r in rows if r['pp_tps'] != 'na']
if len(ok) >= 2:
    base = ok[0]; last = ok[-1]
    dmem = int(last['vram_peak_mib']) - int(base['vram_peak_mib'])
    dpp = (float(last['pp_tps']) / float(base['pp_tps']) - 1) * 100
    dtg = (float(last['tg_tps']) / float(base['tg_tps']) - 1) * 100
    print(f"\n{last['ctk']}/{last['ctv']} 相对 {base['ctk']}/{base['ctv']}："
          f"显存 {dmem:+d} MiB，pp {dpp:+.1f}%，tg {dtg:+.1f}%")
PY
echo
echo "读法：省下来的显存能换多少 ctx？换来之后 pp/tg 掉了多少？这笔账才叫“划算/不划算”。"
