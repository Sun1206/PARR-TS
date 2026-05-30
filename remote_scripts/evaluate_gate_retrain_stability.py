import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "parr_icdm"))

from evaluate_calibrated_conformal import collect_full
from evaluate_deployment_drift_gate import mean_shift_std, top_reduction
from evaluate_score_ensembles import candidate_scores, metric, rank_ensemble
from evaluate_val_calibrated_parr import base_args
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from layers.PARR import PARRPreprocessor


def evaluate_one(case, seed, checkpoint, shift_threshold, min_val_windows):
    cfg = base_args(case)
    exp = Exp_Long_Term_Forecast(cfg)
    exp.model.load_state_dict(torch.load(checkpoint, map_location=exp.device))
    parr = PARRPreprocessor(cfg).eval()
    x_val, y_val, _ = collect_full(exp, parr, "val")
    x_test, y_test, _ = collect_full(exp, parr, "test")
    candidates = candidate_scores(x_val, y_val, x_test)
    for item in candidates.values():
        item["test_spearman"] = metric(item["test"], y_test)

    ranked = sorted(candidates, key=lambda name: candidates[name]["val_spearman"])
    selected = ranked[0]
    negative = [name for name in ranked if candidates[name]["val_spearman"] < 0] or [selected]
    qualities = np.array([-candidates[name]["val_spearman"] for name in negative], dtype=np.float64)
    weights = qualities / (qualities.sum() + 1e-12)
    full_val = rank_ensemble(candidates, negative, "val", weights)
    full_test = rank_ensemble(candidates, negative, "test", weights)

    selected_val = candidates[selected]["val"]
    selected_test = candidates[selected]["test"]
    selected_shift = mean_shift_std(selected_val, selected_test)
    full_shift = mean_shift_std(full_val, full_test)
    shift_reliable = selected_shift <= shift_threshold
    sample_reliable = len(y_val) >= min_val_windows
    deployable = bool(shift_reliable and sample_reliable)
    _, full_reduction = top_reduction(full_test, y_test, 0.25)

    return {
        "case": case,
        "seed": seed,
        "checkpoint": checkpoint,
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "selected": selected,
        "val_spearman": candidates[selected]["val_spearman"],
        "test_spearman": candidates[selected]["test_spearman"],
        "selected_score_ks": float(ks_2samp(selected_val, selected_test).statistic),
        "selected_score_wasserstein": float(wasserstein_distance(selected_val, selected_test)),
        "selected_score_mean_shift_std": selected_shift,
        "full_rank_mean_shift_std": full_shift,
        "shift_reliable": shift_reliable,
        "sample_reliable": sample_reliable,
        "gate": "apply" if deployable else "abstain",
        "full_top25_reduction": full_reduction,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--shift-threshold", type=float, default=0.60)
    parser.add_argument("--min-val-windows", type=int, default=200)
    args = parser.parse_args()

    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            ckpts = sorted(glob.glob(item["checkpoint_glob"]))
            if ckpts:
                row = evaluate_one(
                    item["case"],
                    int(item["seed"]),
                    ckpts[-1],
                    args.shift_threshold,
                    args.min_val_windows,
                )
            else:
                row = {
                    "case": item["case"],
                    "seed": int(item["seed"]),
                    "error": "checkpoint_not_found",
                    "checkpoint_glob": item["checkpoint_glob"],
                }
            rows.append(row)
            print(row, flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
