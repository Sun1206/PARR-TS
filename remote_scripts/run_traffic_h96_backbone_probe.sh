#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library
export PATH=/root/miniconda3/bin:$PATH

if [ ! -s dataset/traffic/traffic.csv ]; then
  mkdir -p dataset/traffic
  curl -L --fail --retry 5 --retry-delay 5 \
    -o dataset/traffic/traffic.csv \
    https://hf-mirror.com/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv
fi

COMMON=(
  --task_name long_term_forecast
  --is_training 1
  --root_path ./dataset/traffic/
  --data_path traffic.csv
  --data custom
  --features M
  --seq_len 96
  --pred_len 96
  --enc_in 862
  --dec_in 862
  --c_out 862
  --num_workers 0
  --train_epochs 3
  --patience 1
  --learning_rate 0.0005
  --batch_size 8
)

python -u run.py "${COMMON[@]}" \
  --label_len 0 \
  --e_layers 3 \
  --d_model 32 \
  --d_ff 64 \
  --n_heads 8 \
  --factor 3 \
  --learning_rate 0.001 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --model TimeMixer \
  --model_id crossdata_timemixer_traffic_96 \
  --des crossdata_timemixer_traffic

python -u run.py "${COMMON[@]}" \
  --label_len 48 \
  --e_layers 2 \
  --d_model 64 \
  --d_ff 128 \
  --n_heads 4 \
  --patch_len 16 \
  --model PatchTST \
  --model_id crossdata_patchtst_traffic_96 \
  --des crossdata_patchtst_traffic

python -u run.py "${COMMON[@]}" \
  --label_len 48 \
  --e_layers 1 \
  --d_model 64 \
  --d_ff 128 \
  --n_heads 4 \
  --factor 3 \
  --patch_len 16 \
  --use_norm 1 \
  --model TimeXer \
  --model_id crossdata_timexer_traffic_96 \
  --des crossdata_timexer_traffic

python -u run.py "${COMMON[@]}" \
  --label_len 48 \
  --e_layers 2 \
  --d_model 64 \
  --d_ff 128 \
  --n_heads 4 \
  --model iTransformer \
  --model_id crossdata_itransformer_traffic_96 \
  --des crossdata_itransformer_traffic

python scripts/parr_icdm/evaluate_selective_risk.py \
  --cases traffic_timemixer traffic_patchtst traffic_timexer traffic_itransformer
python scripts/parr_icdm/evaluate_score_ensembles.py \
  --cases traffic_timemixer traffic_patchtst traffic_timexer traffic_itransformer
python scripts/parr_icdm/evaluate_deployment_drift_gate.py \
  --cases traffic_timemixer traffic_patchtst traffic_timexer traffic_itransformer \
  --shift-threshold 0.60 \
  --min-val-windows 200
