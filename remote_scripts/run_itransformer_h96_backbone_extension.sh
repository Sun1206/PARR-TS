#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library

run_base() {
  local dataset="$1"
  local data_path="$2"
  local model_id="$3"
  local des="$4"
  local glob="$5"
  local batch_size="$6"

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
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 128 \
    --d_ff 256 \
    --n_heads 4 \
    --train_epochs 10 \
    --batch_size "${batch_size}" \
    --num_workers 0 \
    --patience 3 \
    --learning_rate 0.0005 \
    --model iTransformer \
    --model_id "${model_id}" \
    --des "${des}"
}

run_base ETTh1 ETTh1.csv itransformer_base_etth1_96 itransformer_base \
  "checkpoints/long_term_forecast_itransformer_base_etth1_96_iTransformer_ETTh1_*itransformer_base_0/checkpoint.pth" 32
run_base ETTh2 ETTh2.csv crossdata_itransformer_etth2_96 crossdata_itransformer_base \
  "checkpoints/long_term_forecast_crossdata_itransformer_etth2_96_iTransformer_ETTh2_*crossdata_itransformer_base_0/checkpoint.pth" 32
run_base ETTm1 ETTm1.csv crossdata_itransformer_ettm1_96 crossdata_itransformer_base \
  "checkpoints/long_term_forecast_crossdata_itransformer_ettm1_96_iTransformer_ETTm1_*crossdata_itransformer_base_0/checkpoint.pth" 32

python scripts/parr_icdm/evaluate_selective_risk.py \
  --cases etth1_itransformer etth2_itransformer ettm1_itransformer

python scripts/parr_icdm/evaluate_score_ensembles.py \
  --cases etth1_itransformer etth2_itransformer ettm1_itransformer
