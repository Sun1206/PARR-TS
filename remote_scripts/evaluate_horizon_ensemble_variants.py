import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "parr_icdm"))

from evaluate_calibrated_conformal import collect_full
from evaluate_horizon_robustness import (
    candidate_scores,
    make_case,
    rank_ensemble,
    spearman,
    top_reduction,
)
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from layers.PARR import PARRPreprocessor


def weighted_rank(candidates, names):
    names = list(names)
    if not names:
        raise ValueError("weighted_rank requires at least one candidate")
    qualities = np.array([max(0.0, -float(candidates[name]["val_spearman"])) for name in names], dtype=np.float64)
    if qualities.sum() <= 1e-12:
        qualities = np.ones(len(names), dtype=np.float64)
    weights = qualities / qualities.sum()
    return rank_ensemble(candidates, names, weights), dict(zip(names, weights.tolist()))


def evaluate_case(dataset, model, horizon):
    cfg = make_case(dataset, model, horizon)
    ckpts = sorted(glob.glob(cfg.ckpt_glob))
    if not ckpts:
        return {"dataset": dataset, "model": model, "horizon": horizon, "error": "checkpoint_not_found", "glob": cfg.ckpt_glob}

    exp = Exp_Long_Term_Forecast(cfg)
    exp.model.load_state_dict(torch.load(ckpts[-1], map_location=exp.device))
    parr = PARRPreprocessor(cfg).eval()
    x_val, y_val, _ = collect_full(exp, parr, "val")
    x_test, y_test, _ = collect_full(exp, parr, "test")

    candidates = candidate_scores(x_val, y_val, x_test)
    for item in candidates.values():
        item["test_spearman"] = spearman(item["test"], y_test)
    ranked = sorted(candidates, key=lambda name: candidates[name]["val_spearman"])
    negative = [name for name in ranked if candidates[name]["val_spearman"] < 0]
    if not negative:
        negative = [ranked[0]]

    top3_weighted_score, top3_weights = weighted_rank(candidates, negative[:3])
    top5_weighted_score, top5_weights = weighted_rank(candidates, negative[:5])
    full_weighted_score, full_weights = weighted_rank(candidates, negative)
    methods = {
        "hard_selected": candidates[ranked[0]]["test"],
        "top2_rank": rank_ensemble(candidates, ranked[:2]),
        "top3_rank": rank_ensemble(candidates, ranked[:3]),
        "top3_weighted_rank": top3_weighted_score,
        "top5_weighted_rank": top5_weighted_score,
        "negative_weighted_rank": full_weighted_score,
    }

    row = {
        "dataset": dataset,
        "model": model,
        "horizon": int(horizon),
        "overall_mse": float(y_test.mean()),
        "ranked_candidates": ranked,
        "negative_candidates": negative,
        "top3_weights": top3_weights,
        "top5_weights": top5_weights,
        "full_weights": full_weights,
    }
    for name, score in methods.items():
        mse, reduction = top_reduction(score, y_test)
        row[f"{name}_spearman"] = spearman(score, y_test)
        row[f"{name}_top25_mse"] = mse
        row[f"{name}_top25_reduction"] = reduction
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["etth1", "etth2", "ettm1"])
    parser.add_argument("--model", default="timemixer")
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        for horizon in args.horizons:
            row = evaluate_case(dataset, args.model, horizon)
            rows.append(row)
            print(row, flush=True)

    metrics = [
        "hard_selected_top25_reduction",
        "top2_rank_top25_reduction",
        "top3_rank_top25_reduction",
        "top3_weighted_rank_top25_reduction",
        "top5_weighted_rank_top25_reduction",
        "negative_weighted_rank_top25_reduction",
    ]
    summary = {"rows": len([r for r in rows if "error" not in r])}
    valid_rows = [r for r in rows if "error" not in r]
    for metric in metrics:
        values = [float(r[metric]) for r in valid_rows]
        summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
        summary[f"{metric}_min"] = float(np.min(values)) if values else None
        summary[f"{metric}_positive_count"] = int(np.sum(np.array(values) > 0.0)) if values else 0
    print({"summary": summary}, flush=True)


if __name__ == "__main__":
    main()
