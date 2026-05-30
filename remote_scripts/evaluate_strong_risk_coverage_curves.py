import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "parr_icdm"))

from evaluate_calibrated_conformal import collect_full
from evaluate_score_ensembles import candidate_scores, rank_ensemble
from evaluate_strong_risk_score_baselines import (
    DEFAULT_CASES,
    fit_strong_scores,
    metric,
    rank01,
    top_reduction,
)
from evaluate_val_calibrated_parr import base_args
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from layers.PARR import PARRPreprocessor


APPLIED_CASES = [
    case
    for case in DEFAULT_CASES
    if case.startswith(("etth1_", "etth2_", "ettm1_", "ettm2_", "electricity_"))
]


def family(case):
    dataset = case.split("_")[0]
    if dataset.startswith("ett"):
        return "ETT"
    return dataset.capitalize()


def build_scores(x_val, y_val, x_test, seed):
    parr_candidates = candidate_scores(x_val, y_val, x_test)
    ranked = sorted(parr_candidates, key=lambda name: parr_candidates[name]["val_spearman"])
    selected = ranked[0]
    negative = [name for name in ranked if parr_candidates[name]["val_spearman"] < 0] or [selected]
    weights = np.array([-parr_candidates[name]["val_spearman"] for name in negative], dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    scores = {
        "PARR-rank": rank_ensemble(parr_candidates, negative, "test", weights),
    }

    strong = fit_strong_scores(x_val, y_val, x_test, seed)
    for name, (_, test_score) in strong.items():
        if name == "ridge_risk":
            scores["Ridge"] = test_score
        elif name == "gbdt_risk":
            scores["GBDT"] = test_score
        elif name == "mlp_risk":
            scores["MLP"] = test_score
        elif name == "knn_mean_risk":
            scores["kNN mean"] = test_score
        elif name == "knn_q90_risk":
            scores["kNN q90"] = test_score
        elif name == "mahalanobis_ood":
            scores["Mahalanobis OOD"] = test_score
        elif name == "pca_reconstruction_ood":
            scores["PCA OOD"] = test_score
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=APPLIED_CASES)
    parser.add_argument("--coverages", nargs="+", type=float, default=[0.10, 0.25, 0.50, 0.75, 1.00])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--out-jsonl", default="")
    args = parser.parse_args()

    rows = []
    jsonl_handle = None
    if args.out_jsonl:
        os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
        jsonl_handle = open(args.out_jsonl, "w", encoding="utf-8")

    for case in args.cases:
        cfg = base_args(case)
        ckpts = sorted(glob.glob(cfg.ckpt_glob))
        if not ckpts:
            row = {"case": case, "error": "checkpoint_not_found", "glob": cfg.ckpt_glob}
            print(row, flush=True)
            if jsonl_handle:
                jsonl_handle.write(json.dumps(row) + "\n")
                jsonl_handle.flush()
            continue

        exp = Exp_Long_Term_Forecast(cfg)
        exp.model.load_state_dict(torch.load(ckpts[-1], map_location=exp.device))
        parr = PARRPreprocessor(cfg).eval()
        x_val, y_val, _ = collect_full(exp, parr, "val")
        x_test, y_test, _ = collect_full(exp, parr, "test")
        scores = build_scores(x_val, y_val, x_test, args.seed)

        for method, score in scores.items():
            auc_values = []
            for coverage in args.coverages:
                reduction = top_reduction(score, y_test, coverage)
                auc_values.append(reduction)
                row = {
                    "case": case,
                    "family": family(case),
                    "method": method,
                    "coverage": coverage,
                    "reduction": reduction,
                    "n_val": len(y_val),
                    "n_test": len(y_test),
                }
                rows.append(row)
                print(row, flush=True)
                if jsonl_handle:
                    jsonl_handle.write(json.dumps(row) + "\n")
                    jsonl_handle.flush()
            auc_row = {
                "case": case,
                "family": family(case),
                "method": method,
                "coverage": "auc_excl100",
                "reduction": float(np.mean(auc_values[:-1])),
                "n_val": len(y_val),
                "n_test": len(y_test),
            }
            rows.append(auc_row)
            if jsonl_handle:
                jsonl_handle.write(json.dumps(auc_row) + "\n")
                jsonl_handle.flush()

    if jsonl_handle:
        jsonl_handle.close()

    if args.out_csv and rows:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        keys = ["case", "family", "method", "coverage", "reduction", "n_val", "n_test", "error", "glob"]
        with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
