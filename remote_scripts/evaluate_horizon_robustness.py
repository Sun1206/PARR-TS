import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "parr_icdm"))

from evaluate_calibrated_conformal import collect_full, fit_scores_for_val_test
from evaluate_val_calibrated_parr import base_args, sigmoid
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from layers.PARR import PARRPreprocessor


COMPONENTS = ["spectral", "period", "smooth", "channel"]
FIXED_BETA = np.array([-1.0, 1.0, -1.0, -1.0])


def spearman(score, y):
    value = spearmanr(score, y).statistic
    return float(value) if not np.isnan(value) else float("inf")


def rank01(score):
    if len(score) <= 1:
        return np.zeros_like(score, dtype=np.float64)
    return (rankdata(score, method="average") - 1.0) / (len(score) - 1.0)


def top_reduction(score, y, frac=0.25):
    order = np.argsort(score)
    k = max(1, int(frac * len(order)))
    mse = float(y[order[-k:]].mean())
    overall = float(y.mean())
    return mse, float(1.0 - mse / overall)


def make_case(dataset, model, horizon):
    if model != "timemixer" or dataset not in {"etth1", "etth2", "ettm1"}:
        raise ValueError("Only etth1/etth2/ettm1 TimeMixer horizon checkpoints are currently supported.")
    cfg = base_args(f"{dataset}_timemixer")
    cfg.pred_len = int(horizon)
    if dataset == "etth1" and horizon == 96:
        cfg.ckpt_glob = (
            "checkpoints/long_term_forecast_first_real_timemixer_etth1_96_"
            "TimeMixer_ETTh1_*first_real_base_0/checkpoint.pth"
        )
    elif dataset == "etth2" and horizon == 96:
        cfg.ckpt_glob = (
            "checkpoints/long_term_forecast_crossdata_timemixer_etth2_96_"
            "TimeMixer_ETTh2_*crossdata_timemixer_base_0/checkpoint.pth"
        )
    elif dataset == "ettm1" and horizon == 96:
        cfg.ckpt_glob = (
            "checkpoints/long_term_forecast_crossdata_timemixer_ettm1_96_"
            "TimeMixer_ETTm1_*crossdata_timemixer_base_0/checkpoint.pth"
        )
    else:
        data_name = {"etth1": "ETTh1", "etth2": "ETTh2", "ettm1": "ETTm1"}[dataset]
        cfg.ckpt_glob = (
            f"checkpoints/long_term_forecast_predsweep_base_timemixer_{dataset}_{horizon}_"
            f"TimeMixer_{data_name}_*pl{horizon}_*predsweep_base_0/checkpoint.pth"
        )
    return cfg


def candidate_scores(x_val, y_val, x_test):
    mu = x_val.mean(axis=0)
    sigma = x_val.std(axis=0) + 1e-8
    z_val = (x_val - mu) / sigma
    z_test = (x_test - mu) / sigma
    scores = fit_scores_for_val_test(x_val, y_val, x_test)
    candidates = {
        "fixed_etth1": {"val": sigmoid(-(x_val @ FIXED_BETA)), "test": sigmoid(-(x_test @ FIXED_BETA))},
        "sign": {"val": scores["sign"]["val"], "test": scores["sign"]["test"]},
        "ridge": {"val": scores["ridge"]["val"], "test": scores["ridge"]["test"]},
    }
    for i, name in enumerate(COMPONENTS):
        corr_val = spearman(z_val[:, i], y_val)
        direction = 1.0 if np.isinf(corr_val) or corr_val <= 0 else -1.0
        candidates[f"single_{name}"] = {"val": direction * z_val[:, i], "test": direction * z_test[:, i]}
    for item in candidates.values():
        item["val_spearman"] = spearman(item["val"], y_val)
        item["test_spearman"] = None
    return candidates


def rank_ensemble(candidates, names, weights=None):
    if weights is None:
        weights = np.ones(len(names), dtype=np.float64) / len(names)
    ranks = np.vstack([rank01(candidates[name]["test"]) for name in names])
    return weights @ ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="etth1")
    parser.add_argument("--model", default="timemixer")
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    args = parser.parse_args()

    for horizon in args.horizons:
        cfg = make_case(args.dataset, args.model, horizon)
        ckpts = sorted(glob.glob(cfg.ckpt_glob))
        if not ckpts:
            print({"horizon": horizon, "error": "checkpoint_not_found", "glob": cfg.ckpt_glob})
            continue
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
            "dataset": args.dataset,
            "model": args.model,
            "horizon": horizon,
            "checkpoint": ckpts[-1],
            "val_n": int(len(y_val)),
            "test_n": int(len(y_test)),
            "overall_mse": float(y_test.mean()),
            "hard_candidate": hard,
            "oracle_candidate": oracle,
            "ranked_candidates": ranked,
        }
        for name, score in methods.items():
            mse, reduction = top_reduction(score, y_test)
            row[f"{name}_spearman"] = spearman(score, y_test)
            row[f"{name}_top25_mse"] = mse
            row[f"{name}_top25_reduction"] = reduction
        print(row)


if __name__ == "__main__":
    main()
