# PARR-TS Reproduction Bundle

This folder contains the code and paper-side files needed to reproduce the
PARR-TS analyses. Datasets, checkpoints, and large result caches are intentionally
not included.

## Layout

- `local_experiments/`: local diagnostics, SharedLinear, PatchTST-small,
  full-window PatchTST, conservative rank, descriptor selection, utility,
  case-study, effective-sample, Traffic-scale, and overhead scripts.
- `remote_scripts/`: scripts intended to be copied into
  `Time-Series-Library/scripts/parr_icdm/` on a GPU server.


## Local Analyses

Run from the repository root after placing the public datasets under the paths
used by the scripts:

```powershell
python local_experiments/risk_scoring_multiseed.py
python local_experiments/frozen_shared_linear_strong_baselines.py
python local_experiments/rank_omega_diagnostics.py
python local_experiments/conservative_parr_rank.py
python local_experiments/descriptor_selection_audit.py
python local_experiments/parr_case_study_local.py
python local_experiments/measure_overhead_local.py
python local_experiments/effective_sample_diagnostics.py
python local_experiments/traffic_scale_study.py --max-windows 5000 --batch-size 96 --seed 13 --outdir local_experiments/results_traffic_5000
python local_experiments/traffic_frozen_neural_small.py --max-windows 2400 --seed 13 --epochs 8 --patience 2 --batch-size 16 --eval-batch-size 32 --outdir local_experiments/results_traffic_neural_small
python local_experiments/traffic_patchtst_frozen_neural.py --max-windows 2400 --seed 13 --epochs 8 --patience 2 --batch-size 8 --eval-batch-size 16 --outdir local_experiments/results_traffic_patchtst_small
python local_experiments/traffic_patchtst_full_stream.py --seed 13 --epochs 3 --patience 1 --batch-size 2 --eval-batch-size 4 --d-model 512 --d-ff 512 --n-heads 8 --e-layers 2 --lr 0.0001 --outdir local_experiments/results_traffic_patchtst_full --backbone-name PatchTST-full --amp
```

The PatchTST Traffic checks import `models/PatchTST.py` from a
Time-Series-Library checkout placed at the repository root.

## Remote Frozen-Neural Analyses

Copy `remote_scripts/` into the Time-Series-Library working tree as
`scripts/parr_icdm/`, then run:

```bash
python scripts/parr_icdm/evaluate_strong_risk_score_baselines.py \
  --out-csv results/parr_icdm/strong_neural_baselines.csv \
  --out-jsonl results/parr_icdm/strong_neural_baselines.jsonl

python scripts/parr_icdm/evaluate_strong_risk_coverage_curves.py \
  --out-csv results/parr_icdm/risk_coverage_neural_applied.csv \
  --out-jsonl results/parr_icdm/risk_coverage_neural_applied.jsonl

bash scripts/parr_icdm/run_neural_retrain_and_foundation.sh
python scripts/parr_icdm/evaluate_gate_retrain_stability.py \
  --manifest results/parr_icdm/retrain_variance_manifest_20260516_170247.csv \
  --out-csv results/parr_icdm/gate_retrain_stability.csv \
  --shift-threshold 0.60 \
  --min-val-windows 200
```

The Chronos zero-shot foundation script is included, but foundation results are
outside the main paper scope because the original server did not have cached
weights and could not download from Hugging Face during the run.
