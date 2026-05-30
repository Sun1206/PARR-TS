#!/usr/bin/env bash
set -euo pipefail

SCRIPT="$1"
LOG="$2"
THRESHOLD="${3:-35}"
CHECKS="${4:-3}"
SLEEP_SECONDS="${5:-60}"

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library

ok_count=0
while true; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "$(date) gpu_util=${util} mem_used=${mem} ok_count=${ok_count}/${CHECKS}"
  if [[ "${util}" =~ ^[0-9]+$ ]] && (( util <= THRESHOLD )); then
    ok_count=$((ok_count + 1))
  else
    ok_count=0
  fi
  if (( ok_count >= CHECKS )); then
    echo "$(date) starting ${SCRIPT}"
    bash "${SCRIPT}" > "${LOG}" 2>&1
    echo "$(date) finished ${SCRIPT}"
    break
  fi
  sleep "${SLEEP_SECONDS}"
done

