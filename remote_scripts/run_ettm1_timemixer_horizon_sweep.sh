#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library

mkdir -p dataset/ETT-small logs
if [ ! -s dataset/ETT-small/ETTm1.csv ]; then
  python - <<'PY'
from urllib.request import urlretrieve

urlretrieve(
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "dataset/ETT-small/ETTm1.csv",
)
PY
fi

for pred_len in 192 336 720; do
  COMMON=(
    --task_name long_term_forecast
    --is_training 1
    --root_path ./dataset/ETT-small/
    --data_path ETTm1.csv
    --data ETTm1
    --features M
    --seq_len 96
    --label_len 0
    --pred_len "${pred_len}"
    --e_layers 2
    --enc_in 7
    --dec_in 7
    --c_out 7
    --d_model 16
    --d_ff 32
    --down_sampling_layers 2
    --down_sampling_method avg
    --down_sampling_window 2
    --train_epochs 10
    --batch_size 32
    --num_workers 0
    --patience 3
    --learning_rate 0.001
    --model TimeMixer
  )

  python -u run.py "${COMMON[@]}" \
    --model_id "predsweep_base_timemixer_ettm1_${pred_len}" \
    --des "predsweep_base"
done

python scripts/parr_icdm/evaluate_horizon_robustness.py \
  --dataset ettm1 --model timemixer --horizons 96 192 336 720 \
  | tee logs/horizon_robustness_ettm1_timemixer.log
