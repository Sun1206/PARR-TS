#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library

mkdir -p dataset/ETT-small
if [ ! -f dataset/ETT-small/ETTm2.csv ]; then
  python - <<'PY'
from urllib.request import urlretrieve
url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv"
urlretrieve(url, "dataset/ETT-small/ETTm2.csv")
print("downloaded ETTm2.csv")
PY
fi

run_base() {
  local dataset="$1"
  local data_path="$2"
  local model="$3"
  local model_id="$4"
  local des="$5"
  local glob="$6"
  local label_len="$7"
  local e_layers="$8"
  shift 8
  local extra=("$@")

  if compgen -G "${glob}" > /dev/null; then
    echo "$(date) skip existing ${model_id}"
    return
  fi

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path "${data_path}" \
    --data "${dataset}" \
    --features M \
    --seq_len 96 \
    --label_len "${label_len}" \
    --pred_len 96 \
    --e_layers "${e_layers}" \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 128 \
    --d_ff 256 \
    --n_heads 4 \
    --train_epochs 10 \
    --batch_size 32 \
    --num_workers 0 \
    --patience 3 \
    --learning_rate 0.0005 \
    --model "${model}" \
    --model_id "${model_id}" \
    --des "${des}" \
    "${extra[@]}"
}

run_timemixer() {
  local glob="checkpoints/long_term_forecast_crossdata_timemixer_ettm2_96_TimeMixer_ETTm2_*crossdata_timemixer_base_0/checkpoint.pth"
  if compgen -G "${glob}" > /dev/null; then
    echo "$(date) skip existing crossdata_timemixer_ettm2_96"
    return
  fi
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --data ETTm2 \
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
    --n_heads 8 \
    --train_epochs 10 \
    --batch_size 32 \
    --num_workers 0 \
    --patience 3 \
    --learning_rate 0.001 \
    --down_sampling_layers 2 \
    --down_sampling_method avg \
    --down_sampling_window 2 \
    --model TimeMixer \
    --model_id crossdata_timemixer_ettm2_96 \
    --des crossdata_timemixer_base
}

run_timemixer
run_base ETTm2 ETTm2.csv PatchTST crossdata_patchtst_ettm2_96 crossdata_patchtst_base \
  "checkpoints/long_term_forecast_crossdata_patchtst_ettm2_96_PatchTST_ETTm2_*crossdata_patchtst_base_0/checkpoint.pth" 48 2 --patch_len 16
run_base ETTm2 ETTm2.csv TimeXer crossdata_timexer_ettm2_96 crossdata_timexer_base \
  "checkpoints/long_term_forecast_crossdata_timexer_ettm2_96_TimeXer_ETTm2_*crossdata_timexer_base_0/checkpoint.pth" 48 1 --factor 3 --patch_len 16 --use_norm 1
run_base ETTm2 ETTm2.csv iTransformer crossdata_itransformer_ettm2_96 crossdata_itransformer_base \
  "checkpoints/long_term_forecast_crossdata_itransformer_ettm2_96_iTransformer_ETTm2_*crossdata_itransformer_base_0/checkpoint.pth" 48 2

python scripts/parr_icdm/evaluate_selective_risk.py \
  --cases ettm2_timemixer ettm2_patchtst ettm2_timexer ettm2_itransformer

python scripts/parr_icdm/evaluate_score_ensembles.py \
  --cases ettm2_timemixer ettm2_patchtst ettm2_timexer ettm2_itransformer
