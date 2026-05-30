#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library

COMMON=(
  --task_name long_term_forecast
  --is_training 1
  --root_path ./dataset/ETT-small/
  --data_path ETTh1.csv
  --data ETTh1
  --features M
  --seq_len 96
  --label_len 48
  --pred_len 96
  --e_layers 1
  --enc_in 7
  --dec_in 7
  --c_out 7
  --d_model 128
  --d_ff 256
  --n_heads 4
  --factor 3
  --train_epochs 10
  --batch_size 32
  --num_workers 2
  --patience 3
  --learning_rate 0.0005
  --dropout 0.1
  --patch_len 16
  --use_norm 1
  --model TimeXer
)

python -u run.py "${COMMON[@]}" \
  --model_id timexer_base_etth1_96 \
  --des timexer_base

python -u run.py "${COMMON[@]}" \
  --model_id timexer_rawsig_diag_etth1_96 \
  --use_parr \
  --parr_patch_len 16 \
  --parr_alpha_s -1.0 \
  --parr_alpha_d 1.0 \
  --parr_alpha_e -1.0 \
  --parr_alpha_g -1.0 \
  --parr_score_mode sigmoid_raw \
  --parr_dropout 0.0 \
  --parr_replace_strength 0.0 \
  --parr_save_diagnostics \
  --des timexer_rawsig_diag

python -u run.py "${COMMON[@]}" \
  --model_id timexer_rawsig_dropout02_etth1_96 \
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
  --des timexer_rawsig_dropout02

python scripts/parr_icdm/analyze_parr_diagnostics.py | grep -E "timexer_|patchtst_|itransformer_|raw_sigmoid_score|rawsig_dropout02"
