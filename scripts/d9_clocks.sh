#!/usr/bin/env bash
# 压测期间每 0.5 s 采一次 SM 时钟 / 利用率 / 显存，用来判断"慢"是不是降频
DUR=${1:-300}
OUT=${2:-results/d9_clocks.csv}
i=0
echo "ts,clocks_sm_mhz,clocks_max_mhz,util_pct,mem_used_mib,temp_c" > "$OUT"
while [ $i -lt $((DUR*2)) ]; do
  v=$(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  [ -n "$v" ] && echo "$(date +%H:%M:%S),$v" >> "$OUT"
  sleep 0.5
  i=$((i+1))
done