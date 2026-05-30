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


def bootstrap_ci(score, y, frac, boots, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    values = []
    for _ in range(boots):
        idx = rng.integers(0, n, size=n)
        _, reduction = top_reduction(score[idx], y[idx], frac)
        values.append(reduction)
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi), float(np.mean(values))


def evaluate_case(dataset, model, horizon, frac, boots, seed):
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
    hard = ranked[0]
    oracle = min(candidates, key=lambda name: candidates[name]["test_spearman"])
    negative = [name for name in ranked if candidates[name]["val_spearman"] < 0]
    if not negative:
        negative = [hard]
    qualities = np.array([-candidates[name]["val_spearman"] for name in negative], dtype=np.float64)
    weights = qualities / (qualities.sum() + 1e-12)

    methods = {
        "hard_selected": candidates[hard]["test"],
        "top3_rank": rank_ensemble(candidates, ranked[:3]),
        "negative_weighted_rank": rank_ensemble(candidates, negative, weights),
        "oracle_candidate": candidates[oracle]["test"],
    }
    row = {
        "dataset": dataset,
        "model": model,
        "horizon": int(horizon),
        "checkpoint": ckpts[-1],
        "test_n": int(len(y_test)),
        "overall_mse": float(y_test.mean()),
        "hard_candidate": hard,
        "oracle_candidate": oracle,
        "ranked_candidates": ranked,
        "negative_candidates": negative,
    }
    for i, (name, score) in enumerate(methods.items()):
        mse, reduction = top_reduction(score, y_test, frac)
        lo, hi, boot_mean = bootstrap_ci(score, y_test, frac, boots, seed + i)
        row[f"{name}_spearman"] = spearman(score, y_test)
        row[f"{name}_top25_mse"] = mse
        row[f"{name}_top25_reduction"] = reduction
        row[f"{name}_ci95"] = [lo, hi]
        row[f"{name}_boot_mean"] = boot_mean
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["etth1", "etth2", "ettm1"])
    parser.add_argument("--model", default="timemixer")
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--boots", type=int, default=500)
    parser.add_argument("--frac", type=float, default=0.25)
    args = parser.parse_args()

    for d_id, dataset in enumerate(args.datasets):
        for h_id, horizon in enumerate(args.horizons):
            seed = 20260515 + 10000 * d_id + 100 * h_id
            print(evaluate_case(dataset, args.model, horizon, args.frac, args.boots, seed), flush=True)


if __name__ == "__main__":
    main()
