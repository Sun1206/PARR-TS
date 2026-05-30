#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library

COMMON=(
  --task_name long_term_forecast
  --is_training 1
  --root_path ./dataset/ETT-small/
  --data_path ETTh2.csv
  --data ETTh2
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
  --model_id compare_timexer_etth2_96 \
  --des compare_timexer_base

python -u run.py "${COMMON[@]}" \
  --model_id compare_timexer_fixed_rawsig_diag_etth2_96 \
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
  --des compare_timexer_fixed_rawsig_diag

python -u run.py "${COMMON[@]}" \
  --model_id compare_timexer_signcal_rawsig_diag_etth2_96 \
  --use_parr \
  --parr_patch_len 16 \
  --parr_alpha_s 1.0 \
  --parr_alpha_d -1.0 \
  --parr_alpha_e -1.0 \
  --parr_alpha_g 1.0 \
  --parr_score_mode sigmoid_raw \
  --parr_dropout 0.0 \
  --parr_replace_strength 0.0 \
  --parr_save_diagnostics \
  --des compare_timexer_signcal_rawsig_diag

python scripts/parr_icdm/analyze_parr_diagnostics.py | grep -E "compare_timexer|crossdata_|signcal_|timexer_"

for pat in 'results/*compare_timexer_fixed_rawsig_diag_etth2_96*' 'results/*compare_timexer_signcal_rawsig_diag_etth2_96*'; do
  echo "BINS $pat"
  python scripts/parr_icdm/analyze_predictability_bins.py --pattern "$pat" --bins 4
done
