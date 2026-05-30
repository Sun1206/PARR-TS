#!/usr/bin/env bash
set -euo pipefail

SCRIPT="$1"
LOG="$2"
UTIL_THRESHOLD="${3:-20}"
MEM_THRESHOLD="${4:-2500}"
CHECKS="${5:-2}"
SLEEP_SECONDS="${6:-60}"

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library

ok_count=0
while true; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "$(date) gpu_util=${util} mem_used=${mem} ok_count=${ok_count}/${CHECKS} thresholds util<=${UTIL_THRESHOLD} mem<=${MEM_THRESHOLD}"
  if [[ "${util}" =~ ^[0-9]+$ ]] && [[ "${mem}" =~ ^[0-9]+$ ]] && (( util <= UTIL_THRESHOLD )) && (( mem <= MEM_THRESHOLD )); then
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
