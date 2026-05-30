#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library

python scripts/parr_icdm/evaluate_horizon_ensemble_variants.py \
  --datasets etth1 etth2 ettm1 \
  --model timemixer \
  --horizons 96 192 336 720
