#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library
export PATH=/root/miniconda3/bin:$PATH

if [ ! -s dataset/illness/national_illness.csv ]; then
  echo "dataset/illness/national_illness.csv is missing; upload Illness before running this script." >&2
  exit 2
fi

COMMON=(
  --task_name long_term_forecast
  --is_training 1
  --root_path ./dataset/illness/
  --data_path national_illness.csv
  --data custom
  --features M
  --freq w
  --seq_len 36
  --pred_len 24
  --enc_in 7
  --dec_in 7
  --c_out 7
  --num_workers 0
  --train_epochs 10
  --patience 3
  --learning_rate 0.0005
  --batch_size 16
)

python -u run.py "${COMMON[@]}" \
  --label_len 0 \
  --e_layers 2 \
  --d_model 16 \
  --d_ff 32 \
  --n_heads 8 \
  --factor 3 \
  --learning_rate 0.001 \
  --batch_size 32 \
  --down_sampling_layers 2 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --model TimeMixer \
  --model_id crossdata_timemixer_illness_24 \
  --des crossdata_timemixer_illness

python -u run.py "${COMMON[@]}" \
  --label_len 18 \
  --e_layers 2 \
  --d_model 64 \
  --d_ff 128 \
  --n_heads 4 \
  --patch_len 16 \
  --model PatchTST \
  --model_id crossdata_patchtst_illness_24 \
  --des crossdata_patchtst_illness

python -u run.py "${COMMON[@]}" \
  --label_len 18 \
  --e_layers 1 \
  --d_model 64 \
  --d_ff 128 \
  --n_heads 4 \
  --factor 3 \
  --patch_len 16 \
  --use_norm 1 \
  --model TimeXer \
  --model_id crossdata_timexer_illness_24 \
  --des crossdata_timexer_illness

python -u run.py "${COMMON[@]}" \
  --label_len 18 \
  --e_layers 2 \
  --d_model 64 \
  --d_ff 128 \
  --n_heads 4 \
  --model iTransformer \
  --model_id crossdata_itransformer_illness_24 \
  --des crossdata_itransformer_illness

python scripts/parr_icdm/evaluate_selective_risk.py \
  --cases illness_timemixer illness_patchtst illness_timexer illness_itransformer
python scripts/parr_icdm/evaluate_score_ensembles.py \
  --cases illness_timemixer illness_patchtst illness_timexer illness_itransformer
python scripts/parr_icdm/evaluate_deployment_drift_gate.py \
  --cases illness_timemixer illness_patchtst illness_timexer illness_itransformer \
  --shift-threshold 0.60
