#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library
export PATH=/root/miniconda3/bin:$PATH

if [ ! -s dataset/weather/weather.csv ]; then
  echo "dataset/weather/weather.csv is missing; upload Weather before running this script." >&2
  exit 2
fi

COMMON=(
  --task_name long_term_forecast
  --is_training 1
  --root_path ./dataset/weather/
  --data_path weather.csv
  --data custom
  --features M
  --seq_len 96
  --pred_len 96
  --enc_in 21
  --dec_in 21
  --c_out 21
  --num_workers 0
  --train_epochs 10
  --patience 3
  --learning_rate 0.0005
  --batch_size 32
)

python -u run.py "${COMMON[@]}" \
  --label_len 0 \
  --e_layers 3 \
  --d_model 16 \
  --d_ff 32 \
  --n_heads 8 \
  --factor 3 \
  --learning_rate 0.001 \
  --batch_size 128 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --model TimeMixer \
  --model_id crossdata_timemixer_weather_96 \
  --des crossdata_timemixer_weather

python -u run.py "${COMMON[@]}" \
  --label_len 48 \
  --e_layers 2 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 4 \
  --patch_len 16 \
  --model PatchTST \
  --model_id crossdata_patchtst_weather_96 \
  --des crossdata_patchtst_weather

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
  --model_id crossdata_timexer_weather_96 \
  --des crossdata_timexer_weather

python -u run.py "${COMMON[@]}" \
  --label_len 48 \
  --e_layers 2 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 4 \
  --model iTransformer \
  --model_id crossdata_itransformer_weather_96 \
  --des crossdata_itransformer_weather

python scripts/parr_icdm/evaluate_selective_risk.py \
  --cases weather_timemixer weather_patchtst weather_timexer weather_itransformer
python scripts/parr_icdm/evaluate_score_ensembles.py \
  --cases weather_timemixer weather_patchtst weather_timexer weather_itransformer
