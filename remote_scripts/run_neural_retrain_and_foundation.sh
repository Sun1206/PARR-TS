#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/parr_ts_icdm/Time-Series-Library

mkdir -p logs/parr_icdm results/parr_icdm dataset/ETT-small

if [ ! -f dataset/ETT-small/ETTm2.csv ]; then
  python - <<'PY'
from urllib.request import urlretrieve
urlretrieve("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv", "dataset/ETT-small/ETTm2.csv")
PY
fi

if ! grep -q 'TSLIB_FIX_SEED' run.py; then
  python - <<'PY'
from pathlib import Path
p = Path("run.py")
s = p.read_text()
s = s.replace("fix_seed = 2021", 'fix_seed = int(os.environ.get("TSLIB_FIX_SEED", "2021"))')
p.write_text(s)
PY
fi

run_if_missing() {
  local seed="$1"; shift
  local glob="$1"; shift
  if compgen -G "$glob" > /dev/null; then
    echo "$(date) skip existing seed=${seed} glob=${glob}"
    return
  fi
  echo "$(date) train seed=${seed}: $*"
  TSLIB_FIX_SEED="$seed" python -u run.py "$@"
}

train_ettm2() {
  local seed="$1"
  run_if_missing "$seed" "checkpoints/long_term_forecast_parrseed${seed}_ettm2_timemixer_96_TimeMixer_ETTm2_*parrseed${seed}_0/checkpoint.pth" \
    --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --data ETTm2 --features M \
    --seq_len 96 --label_len 0 --pred_len 96 --e_layers 2 \
    --enc_in 7 --dec_in 7 --c_out 7 --d_model 16 --d_ff 32 --n_heads 8 \
    --train_epochs 10 --batch_size 512 --num_workers 0 --patience 3 \
    --learning_rate 0.001 --down_sampling_layers 2 --down_sampling_method avg --down_sampling_window 2 \
    --model TimeMixer --model_id parrseed${seed}_ettm2_timemixer_96 --des parrseed${seed}

  run_if_missing "$seed" "checkpoints/long_term_forecast_parrseed${seed}_ettm2_patchtst_96_PatchTST_ETTm2_*parrseed${seed}_0/checkpoint.pth" \
    --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --data ETTm2 --features M \
    --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 \
    --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --n_heads 4 \
    --train_epochs 10 --batch_size 512 --num_workers 0 --patience 3 \
    --learning_rate 0.0005 --patch_len 16 \
    --model PatchTST --model_id parrseed${seed}_ettm2_patchtst_96 --des parrseed${seed}

  run_if_missing "$seed" "checkpoints/long_term_forecast_parrseed${seed}_ettm2_timexer_96_TimeXer_ETTm2_*parrseed${seed}_0/checkpoint.pth" \
    --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --data ETTm2 --features M \
    --seq_len 96 --label_len 48 --pred_len 96 --e_layers 1 \
    --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --n_heads 4 \
    --train_epochs 10 --batch_size 512 --num_workers 0 --patience 3 \
    --learning_rate 0.0005 --factor 3 --patch_len 16 --use_norm 1 \
    --model TimeXer --model_id parrseed${seed}_ettm2_timexer_96 --des parrseed${seed}

  run_if_missing "$seed" "checkpoints/long_term_forecast_parrseed${seed}_ettm2_itransformer_96_iTransformer_ETTm2_*parrseed${seed}_0/checkpoint.pth" \
    --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --data ETTm2 --features M \
    --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 \
    --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --n_heads 4 \
    --train_epochs 10 --batch_size 512 --num_workers 0 --patience 3 \
    --learning_rate 0.0005 \
    --model iTransformer --model_id parrseed${seed}_ettm2_itransformer_96 --des parrseed${seed}
}

train_electricity() {
  local seed="$1"
  if [ ! -s dataset/electricity/electricity.csv ]; then
    echo "dataset/electricity/electricity.csv missing; skipping Electricity seed ${seed}" >&2
    return
  fi
  local common=(--task_name long_term_forecast --is_training 1
    --root_path ./dataset/electricity/ --data_path electricity.csv --data custom --features M
    --seq_len 96 --pred_len 96 --enc_in 321 --dec_in 321 --c_out 321
    --num_workers 0 --train_epochs 5 --patience 2 --learning_rate 0.0005 --batch_size 16)

  run_if_missing "$seed" "checkpoints/long_term_forecast_parrseed${seed}_electricity_timemixer_96_TimeMixer_custom_*parrseed${seed}_0/checkpoint.pth" \
    "${common[@]}" --label_len 0 --e_layers 2 --d_model 16 --d_ff 32 --n_heads 8 --factor 3 \
    --learning_rate 0.001 --batch_size 32 --down_sampling_layers 2 --down_sampling_method avg --down_sampling_window 2 \
    --model TimeMixer --model_id parrseed${seed}_electricity_timemixer_96 --des parrseed${seed}

  run_if_missing "$seed" "checkpoints/long_term_forecast_parrseed${seed}_electricity_patchtst_96_PatchTST_custom_*parrseed${seed}_0/checkpoint.pth" \
    "${common[@]}" --label_len 48 --e_layers 2 --d_model 64 --d_ff 128 --n_heads 4 --patch_len 16 \
    --model PatchTST --model_id parrseed${seed}_electricity_patchtst_96 --des parrseed${seed}

  run_if_missing "$seed" "checkpoints/long_term_forecast_parrseed${seed}_electricity_timexer_96_TimeXer_custom_*parrseed${seed}_0/checkpoint.pth" \
    "${common[@]}" --label_len 48 --e_layers 1 --d_model 64 --d_ff 128 --n_heads 4 --factor 3 --patch_len 16 --use_norm 1 \
    --model TimeXer --model_id parrseed${seed}_electricity_timexer_96 --des parrseed${seed}

  run_if_missing "$seed" "checkpoints/long_term_forecast_parrseed${seed}_electricity_itransformer_96_iTransformer_custom_*parrseed${seed}_0/checkpoint.pth" \
    "${common[@]}" --label_len 48 --e_layers 2 --d_model 64 --d_ff 128 --n_heads 4 \
    --model iTransformer --model_id parrseed${seed}_electricity_itransformer_96 --des parrseed${seed}
}

for seed in 2022 2023; do
  train_ettm2 "$seed"
  train_electricity "$seed"
done

MANIFEST=results/parr_icdm/retrain_variance_manifest_$(date +%Y%m%d_%H%M%S).csv
OUT=results/parr_icdm/retrain_variance_keycases_$(date +%Y%m%d_%H%M%S).csv
cat > "$MANIFEST" <<'EOF'
case,seed,checkpoint_glob
ettm2_timemixer,2021,checkpoints/long_term_forecast_crossdata_timemixer_ettm2_96_TimeMixer_ETTm2_*crossdata_timemixer_base_0/checkpoint.pth
ettm2_patchtst,2021,checkpoints/long_term_forecast_crossdata_patchtst_ettm2_96_PatchTST_ETTm2_*crossdata_patchtst_base_0/checkpoint.pth
ettm2_timexer,2021,checkpoints/long_term_forecast_crossdata_timexer_ettm2_96_TimeXer_ETTm2_*crossdata_timexer_base_0/checkpoint.pth
ettm2_itransformer,2021,checkpoints/long_term_forecast_crossdata_itransformer_ettm2_96_iTransformer_ETTm2_*crossdata_itransformer_base_0/checkpoint.pth
electricity_timemixer,2021,checkpoints/long_term_forecast_crossdata_timemixer_electricity_96_TimeMixer_custom_*crossdata_timemixer_electricity_0/checkpoint.pth
electricity_patchtst,2021,checkpoints/long_term_forecast_crossdata_patchtst_electricity_96_PatchTST_custom_*crossdata_patchtst_electricity_0/checkpoint.pth
electricity_timexer,2021,checkpoints/long_term_forecast_crossdata_timexer_electricity_96_TimeXer_custom_*crossdata_timexer_electricity_0/checkpoint.pth
electricity_itransformer,2021,checkpoints/long_term_forecast_crossdata_itransformer_electricity_96_iTransformer_custom_*crossdata_itransformer_electricity_0/checkpoint.pth
ettm2_timemixer,2022,checkpoints/long_term_forecast_parrseed2022_ettm2_timemixer_96_TimeMixer_ETTm2_*parrseed2022_0/checkpoint.pth
ettm2_patchtst,2022,checkpoints/long_term_forecast_parrseed2022_ettm2_patchtst_96_PatchTST_ETTm2_*parrseed2022_0/checkpoint.pth
ettm2_timexer,2022,checkpoints/long_term_forecast_parrseed2022_ettm2_timexer_96_TimeXer_ETTm2_*parrseed2022_0/checkpoint.pth
ettm2_itransformer,2022,checkpoints/long_term_forecast_parrseed2022_ettm2_itransformer_96_iTransformer_ETTm2_*parrseed2022_0/checkpoint.pth
electricity_timemixer,2022,checkpoints/long_term_forecast_parrseed2022_electricity_timemixer_96_TimeMixer_custom_*parrseed2022_0/checkpoint.pth
electricity_patchtst,2022,checkpoints/long_term_forecast_parrseed2022_electricity_patchtst_96_PatchTST_custom_*parrseed2022_0/checkpoint.pth
electricity_timexer,2022,checkpoints/long_term_forecast_parrseed2022_electricity_timexer_96_TimeXer_custom_*parrseed2022_0/checkpoint.pth
electricity_itransformer,2022,checkpoints/long_term_forecast_parrseed2022_electricity_itransformer_96_iTransformer_custom_*parrseed2022_0/checkpoint.pth
ettm2_timemixer,2023,checkpoints/long_term_forecast_parrseed2023_ettm2_timemixer_96_TimeMixer_ETTm2_*parrseed2023_0/checkpoint.pth
ettm2_patchtst,2023,checkpoints/long_term_forecast_parrseed2023_ettm2_patchtst_96_PatchTST_ETTm2_*parrseed2023_0/checkpoint.pth
ettm2_timexer,2023,checkpoints/long_term_forecast_parrseed2023_ettm2_timexer_96_TimeXer_ETTm2_*parrseed2023_0/checkpoint.pth
ettm2_itransformer,2023,checkpoints/long_term_forecast_parrseed2023_ettm2_itransformer_96_iTransformer_ETTm2_*parrseed2023_0/checkpoint.pth
electricity_timemixer,2023,checkpoints/long_term_forecast_parrseed2023_electricity_timemixer_96_TimeMixer_custom_*parrseed2023_0/checkpoint.pth
electricity_patchtst,2023,checkpoints/long_term_forecast_parrseed2023_electricity_patchtst_96_PatchTST_custom_*parrseed2023_0/checkpoint.pth
electricity_timexer,2023,checkpoints/long_term_forecast_parrseed2023_electricity_timexer_96_TimeXer_custom_*parrseed2023_0/checkpoint.pth
electricity_itransformer,2023,checkpoints/long_term_forecast_parrseed2023_electricity_itransformer_96_iTransformer_custom_*parrseed2023_0/checkpoint.pth
EOF

python scripts/parr_icdm/evaluate_retrain_variance.py --manifest "$MANIFEST" --out-csv "$OUT"
echo "RETRAIN_VARIANCE_OUT=$OUT"

FOUND_OUT=results/parr_icdm/foundation_chronos_ettm2_$(date +%Y%m%d_%H%M%S).csv
echo "$(date) installing Chronos foundation dependencies if needed"
if python - <<'PY'
import chronos
PY
then
  echo "chronos already installed"
else
  pip install -q chronos-forecasting transformers accelerate
fi

set +e
python scripts/parr_icdm/evaluate_foundation_zero_shot_parr.py \
  --model Chronos \
  --dataset ETTm2 \
  --out-csv "$FOUND_OUT"
FOUND_RC=$?
set -e
echo "FOUNDATION_OUT=$FOUND_OUT"
echo "FOUNDATION_RC=$FOUND_RC"
exit 0
