#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library

MODELS=("TimeMixer" "iTransformer" "PatchTST" "DLinear")
PREDS=(96 192 336 720)

for model in "${MODELS[@]}"; do
  for pred_len in "${PREDS[@]}"; do
    common=(
      --task_name long_term_forecast
      --is_training 1
      --root_path ./dataset/ETT-small/
      --data_path ETTh1.csv
      --data ETTh1
      --features M
      --seq_len 96
      --label_len 48
      --pred_len "${pred_len}"
      --enc_in 7
      --dec_in 7
      --c_out 7
      --train_epochs 10
      --batch_size 32
      --num_workers 2
      --patience 3
      --learning_rate 0.0005
    )

    extra=()
    if [[ "${model}" == "TimeMixer" ]]; then
      extra=(--label_len 0 --d_model 16 --d_ff 32 --e_layers 2 --down_sampling_layers 2 --down_sampling_method avg --down_sampling_window 2)
    elif [[ "${model}" == "iTransformer" ]]; then
      extra=(--d_model 128 --d_ff 256 --e_layers 2 --n_heads 4)
    elif [[ "${model}" == "PatchTST" ]]; then
      extra=(--d_model 128 --d_ff 256 --e_layers 2 --n_heads 4 --patch_len 16)
    fi

    python -u run.py "${common[@]}" "${extra[@]}" \
      --model_id "core_${model}_pl${pred_len}" \
      --model "${model}" \
      --des "core_base"

    python -u run.py "${common[@]}" "${extra[@]}" \
      --model_id "parr_${model}_pl${pred_len}" \
      --model "${model}" \
      --use_parr \
      --parr_patch_len 16 \
      --parr_dropout 0.2 \
      --parr_weighted_loss \
      --parr_save_diagnostics \
      --des "core_parr"
  done
done

