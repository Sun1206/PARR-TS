#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library

PREDS=(192 336 720)

for pred_len in "${PREDS[@]}"; do
  COMMON=(
    --task_name long_term_forecast
    --is_training 1
    --root_path ./dataset/ETT-small/
    --data_path ETTh1.csv
    --data ETTh1
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
    --num_workers 2
    --patience 3
    --learning_rate 0.001
    --model TimeMixer
  )

  python -u run.py "${COMMON[@]}" \
    --model_id "predsweep_base_timemixer_etth1_${pred_len}" \
    --des "predsweep_base"

  python -u run.py "${COMMON[@]}" \
    --model_id "predsweep_rawsig_dropout02_timemixer_etth1_${pred_len}" \
    --use_parr \
    --parr_patch_len 16 \
    --parr_alpha_s -1.0 \
    --parr_alpha_d 1.0 \
    --parr_alpha_e -1.0 \
    --parr_alpha_g -1.0 \
    --parr_score_mode sigmoid_raw \
    --parr_dropout 0.2 \
    --parr_replace_strength 0.0 \
    --parr_save_diagnostics \
    --des "predsweep_rawsig_dropout02"
done

python scripts/parr_icdm/analyze_parr_diagnostics.py | grep -E "predsweep_|rawsig_dropout02|first_real"

