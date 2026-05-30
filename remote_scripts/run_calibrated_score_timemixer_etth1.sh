#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library

# Direction-calibrated diagnostic run:
# component analysis on the baseline showed spectral entropy, smooth residual,
# and channel profile drift are negatively correlated with residual on ETTh1,
# while period drift is weakly positively correlated.
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --down_sampling_layers 2 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --train_epochs 10 \
  --batch_size 32 \
  --num_workers 2 \
  --patience 3 \
  --learning_rate 0.001 \
  --model_id calibrated_score_timemixer_etth1_96 \
  --model TimeMixer \
  --use_parr \
  --parr_patch_len 16 \
  --parr_alpha_s -1.0 \
  --parr_alpha_d 1.0 \
  --parr_alpha_e -1.0 \
  --parr_alpha_g -1.0 \
  --parr_score_mode sigmoid \
  --parr_dropout 0.0 \
  --parr_replace_strength 0.0 \
  --parr_save_diagnostics \
  --des calibrated_score

python scripts/parr_icdm/analyze_parr_diagnostics.py | grep -E "first_real|diag_only|calibrated_score"

