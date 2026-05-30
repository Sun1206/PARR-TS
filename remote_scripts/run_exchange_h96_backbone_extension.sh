#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library
export PATH=/root/miniconda3/bin:$PATH

if [ ! -s dataset/exchange_rate/exchange_rate.csv ]; then
  echo "dataset/exchange_rate/exchange_rate.csv is missing; upload Exchange before running this script." >&2
  exit 2
fi

COMMON=(
  --task_name long_term_forecast
  --is_training 1
  --root_path ./dataset/exchange_rate/
  --data_path exchange_rate.csv
  --data custom
  --features M
  --seq_len 96
  --pred_len 96
  --enc_in 8
  --dec_in 8
  --c_out 8
  --num_workers 0
  --train_epochs 10
  --patience 3
  --learning_rate 0.0005
  --batch_size 32
)

python -u run.py "${COMMON[@]}" \
  --label_len 0 \
  --e_layers 2 \
  --d_model 16 \
  --d_ff 32 \
  --n_heads 8 \
  --factor 3 \
  --learning_rate 0.001 \
  --batch_size 64 \
  --down_sampling_layers 2 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --model TimeMixer \
  --model_id crossdata_timemixer_exchange_96 \
  --des crossdata_timemixer_exchange

python -u run.py "${COMMON[@]}" \
  --label_len 48 \
  --e_layers 2 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 4 \
  --patch_len 16 \
  --model PatchTST \
  --model_id crossdata_patchtst_exchange_96 \
  --des crossdata_patchtst_exchange

python -u run.py "${COMMON[@]}" \
  --label_len 48 \
  --e_layers 1 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 4 \
  --factor 3 \
  --patch_len 16 \
  --use_norm 1 \
  --model TimeXer \
  --model_id crossdata_timexer_exchange_96 \
  --des crossdata_timexer_exchange

python -u run.py "${COMMON[@]}" \
  --label_len 48 \
  --e_layers 2 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 4 \
  --model iTransformer \
  --model_id crossdata_itransformer_exchange_96 \
  --des crossdata_itransformer_exchange

python scripts/parr_icdm/evaluate_selective_risk.py \
  --cases exchange_timemixer exchange_patchtst exchange_timexer exchange_itransformer
python scripts/parr_icdm/evaluate_score_ensembles.py \
  --cases exchange_timemixer exchange_patchtst exchange_timexer exchange_itransformer
python scripts/parr_icdm/evaluate_deployment_drift_gate.py \
  --cases exchange_timemixer exchange_patchtst exchange_timexer exchange_itransformer \
  --shift-threshold 0.60
