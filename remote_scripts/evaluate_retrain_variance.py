import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "parr_icdm"))

from evaluate_calibrated_conformal import collect_full
from evaluate_score_ensembles import candidate_scores, rank_ensemble
from evaluate_strong_risk_score_baselines import fit_strong_scores, rank01, top_reduction
from evaluate_val_calibrated_parr import base_args
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from layers.PARR import PARRPreprocessor


def metric(score, residual):
    value = spearmanr(score, residual).statistic
    return float(value) if not np.isnan(value) else 0.0


def coverage_auc(score, residual, coverages=(0.10, 0.25, 0.50, 0.75)):
    return float(np.mean([top_reduction(score, residual, c) for c in coverages]))


def parr_rank_score(x_val, y_val, x_test):
    candidates = candidate_scores(x_val, y_val, x_test)
    ranked = sorted(candidates, key=lambda name: candidates[name]["val_spearman"])
    selected = ranked[0]
    negative = [name for name in ranked if candidates[name]["val_spearman"] < 0] or [selected]
    weights = np.array([-candidates[name]["val_spearman"] for name in negative], dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    return rank_ensemble(candidates, negative, "test", weights), selected, ",".join(negative)


def evaluate_one(case, checkpoint, seed):
    cfg = base_args(case)
    exp = Exp_Long_Term_Forecast(cfg)
    exp.model.load_state_dict(torch.load(checkpoint, map_location=exp.device))
    parr = PARRPreprocessor(cfg).eval()
    x_val, y_val, _ = collect_full(exp, parr, "val")
    x_test, y_test, _ = collect_full(exp, parr, "test")

    parr_score, selected, negative_pool = parr_rank_score(x_val, y_val, x_test)
    strong = fit_strong_scores(x_val, y_val, x_test, seed)
    ridge = strong["ridge_risk"][1]
    gbdt = strong["gbdt_risk"][1]
    maha = strong["mahalanobis_ood"][1]

    return {
        "case": case,
        "seed": seed,
        "checkpoint": checkpoint,
        "n_val": len(y_val),
        "n_test": len(y_test),
        "overall_mse": float(y_test.mean()),
        "selected": selected,
        "negative_pool": negative_pool,
        "parr_top10": top_reduction(parr_score, y_test, 0.10),
        "parr_top25": top_reduction(parr_score, y_test, 0.25),
        "parr_top50": top_reduction(parr_score, y_test, 0.50),
        "parr_auc": coverage_auc(parr_score, y_test),
        "parr_spearman": metric(parr_score, y_test),
        "ridge_top25": top_reduction(ridge, y_test, 0.25),
        "gbdt_top25": top_reduction(gbdt, y_test, 0.25),
        "maha_top25": top_reduction(maha, y_test, 0.25),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV with columns case,seed,checkpoint_glob")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            case = item["case"]
            seed = int(item["seed"])
            ckpts = sorted(glob.glob(item["checkpoint_glob"]))
            if not ckpts:
                row = {
                    "case": case,
                    "seed": seed,
                    "error": "checkpoint_not_found",
                    "checkpoint_glob": item["checkpoint_glob"],
                }
            else:
                row = evaluate_one(case, ckpts[-1], seed)
            print(row, flush=True)
            rows.append(row)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
