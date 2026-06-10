#!/usr/bin/env bash
# Gate for shared-GPU coordination: exit 0 only when it is safe to launch
# training. Never touches other processes.
#
# Phase 1 (up to 4 h): poll once/min for the sentinel flag dropped by the
#   benchmark session when the GPU is fully free.
# Phase 2 (fallback if the flag never appears): poll nvidia-smi once/min and
#   require memory.used < 2000 MiB for 5 CONSECUTIVE checks (usage dips
#   briefly between benchmark jobs, so one low reading is not enough).

FLAG="/c/Code/AI Research/projects/25-lodestone-verifier-prior/bench/GPU_FREE.flag"

for i in $(seq 1 240); do
  if [ -f "$FLAG" ]; then
    echo "FLAG_FOUND after ~${i} min"
    exit 0
  fi
  sleep 60
done

echo "FLAG_TIMEOUT after 4 h; falling back to nvidia-smi gating"
consec=0
while true; do
  if [ -f "$FLAG" ]; then
    echo "FLAG_FOUND (late, during fallback)"
    exit 0
  fi
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')
  if [ -n "$mem" ] && [ "$mem" -lt 2000 ] 2>/dev/null; then
    consec=$((consec + 1))
    echo "low VRAM ${mem} MiB (${consec}/5)"
  else
    consec=0
  fi
  if [ "$consec" -ge 5 ]; then
    echo "GPU_FREE_CONFIRMED (5 consecutive sub-2000 MiB readings)"
    exit 0
  fi
  sleep 60
done
