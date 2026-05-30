#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 4 \
  --factor 3 \
  --train_epochs 10 \
  --batch_size 64 \
  --num_workers 2 \
  --patience 3 \
  --learning_rate 0.0005 \
  --dropout 0.1 \
  --patch_len 16 \
  --use_norm 1 \
  --model TimeXer \
  --model_id crossdata_timexer_ettm1_96 \
  --des crossdata_timexer_base

python scripts/parr_icdm/evaluate_val_calibrated_parr.py --cases ettm1_timexer | tee logs/val_calibrated_parr_ettm1_timexer_eval.log
