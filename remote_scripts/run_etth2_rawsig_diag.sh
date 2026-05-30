#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library

mkdir -p dataset/ETT-small
if [ ! -s dataset/ETT-small/ETTh2.csv ]; then
  python - <<'PY'
from urllib.request import urlretrieve

urlretrieve(
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "dataset/ETT-small/ETTh2.csv",
)
PY
fi

PARR=(
  --use_parr
  --parr_patch_len 16
  --parr_alpha_s -1.0
  --parr_alpha_d 1.0
  --parr_alpha_e -1.0
  --parr_alpha_g -1.0
  --parr_score_mode sigmoid_raw
  --parr_dropout 0.0
  --parr_replace_strength 0.0
  --parr_save_diagnostics
)

TIMEMIXER_COMMON=(
  --task_name long_term_forecast
  --is_training 1
  --root_path ./dataset/ETT-small/
  --data_path ETTh2.csv
  --data ETTh2
  --features M
  --seq_len 96
  --label_len 0
  --pred_len 96
  --e_layers 2
  --enc_in 7
  --dec_in 7
  --c_out 7
  --d_model 16
  --d_ff 32
  --down_sampling_layers 2
  --down_sampling_method avg
  --down_sampling_window 2
  --train_epochs 10
  --batch_size 32
  --num_workers 2
  --patience 3
  --learning_rate 0.001
  --model TimeMixer
)

PATCHTST_COMMON=(
  --task_name long_term_forecast
  --is_training 1
  --root_path ./dataset/ETT-small/
  --data_path ETTh2.csv
  --data ETTh2
  --features M
  --seq_len 96
  --label_len 48
  --pred_len 96
  --e_layers 2
  --enc_in 7
  --dec_in 7
  --c_out 7
  --d_model 128
  --d_ff 256
  --n_heads 4
  --train_epochs 10
  --batch_size 32
  --num_workers 2
  --patience 3
  --learning_rate 0.0005
  --dropout 0.1
  --patch_len 16
  --model PatchTST
)

python -u run.py "${TIMEMIXER_COMMON[@]}" \
  --model_id crossdata_timemixer_etth2_96 \
  --des crossdata_timemixer_base

python -u run.py "${TIMEMIXER_COMMON[@]}" "${PARR[@]}" \
  --model_id crossdata_timemixer_rawsig_diag_etth2_96 \
  --des crossdata_timemixer_rawsig_diag

python -u run.py "${PATCHTST_COMMON[@]}" \
  --model_id crossdata_patchtst_etth2_96 \
  --des crossdata_patchtst_base

python -u run.py "${PATCHTST_COMMON[@]}" "${PARR[@]}" \
  --model_id crossdata_patchtst_rawsig_diag_etth2_96 \
  --des crossdata_patchtst_rawsig_diag

python scripts/parr_icdm/analyze_parr_diagnostics.py | grep -E "crossdata_|timexer_|patchtst_|itransformer_|raw_sigmoid_score"

for pat in 'results/*crossdata_timemixer_rawsig_diag_etth2_96*' 'results/*crossdata_patchtst_rawsig_diag_etth2_96*'; do
  echo "BINS $pat"
  python scripts/parr_icdm/analyze_predictability_bins.py --pattern "$pat" --bins 4
done
