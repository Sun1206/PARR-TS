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


def metric(score, y):
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
        corr_val = metric(z_val[:, i], y_val)
        direction = 1.0 if np.isinf(corr_val) or corr_val <= 0 else -1.0
        candidates[f"single_{name}"] = {"val": direction * z_val[:, i], "test": direction * z_test[:, i]}
    for item in candidates.values():
        item["val_spearman"] = metric(item["val"], y_val)
    return candidates


def temporal_folds(n, folds):
    edges = np.linspace(0, n, folds + 1, dtype=int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(folds)]


def fold_winners(x_val, y_val, folds=4):
    winners = []
    all_idx = np.arange(len(y_val))
    for hold_idx in temporal_folds(len(y_val), folds):
        train_idx = np.setdiff1d(all_idx, hold_idx, assume_unique=True)
        candidates = candidate_scores(x_val[train_idx], y_val[train_idx], x_val[hold_idx])
        hold_metrics = {name: metric(item["test"], y_val[hold_idx]) for name, item in candidates.items()}
        winners.append(min(hold_metrics, key=hold_metrics.get))
    return winners


def rank_ensemble(candidates, names, split, weights=None):
    if weights is None:
        weights = np.ones(len(names), dtype=np.float64) / len(names)
    ranks = np.vstack([rank01(candidates[name][split]) for name in names])
    return weights @ ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "etth1_timemixer",
            "etth1_patchtst",
            "etth1_timexer",
            "etth1_itransformer",
            "etth2_timemixer",
            "etth2_patchtst",
            "etth2_timexer",
            "etth2_itransformer",
            "ettm1_timemixer",
            "ettm1_patchtst",
            "ettm1_timexer",
            "ettm1_itransformer",
            "ettm2_timemixer",
            "ettm2_patchtst",
            "ettm2_timexer",
            "ettm2_itransformer",
            "weather_timemixer",
            "weather_patchtst",
            "weather_timexer",
            "weather_itransformer",
            "exchange_timemixer",
            "exchange_patchtst",
            "exchange_timexer",
            "exchange_itransformer",
            "electricity_timemixer",
            "electricity_patchtst",
            "electricity_timexer",
            "electricity_itransformer",
            "traffic_timemixer",
            "traffic_patchtst",
            "traffic_timexer",
            "traffic_itransformer",
            "illness_timemixer",
            "illness_patchtst",
            "illness_timexer",
            "illness_itransformer",
        ],
    )
    args = parser.parse_args()

    for case in args.cases:
        cfg = base_args(case)
        ckpts = sorted(glob.glob(cfg.ckpt_glob))
        if not ckpts:
            print({"case": case, "error": "checkpoint_not_found", "glob": cfg.ckpt_glob})
            continue
        exp = Exp_Long_Term_Forecast(cfg)
        exp.model.load_state_dict(torch.load(ckpts[-1], map_location=exp.device))
        parr = PARRPreprocessor(cfg).eval()
        x_val, y_val, _ = collect_full(exp, parr, "val")
        x_test, y_test, _ = collect_full(exp, parr, "test")
        candidates = candidate_scores(x_val, y_val, x_test)

        for item in candidates.values():
            item["test_spearman"] = metric(item["test"], y_test)
        ranked = sorted(candidates, key=lambda name: candidates[name]["val_spearman"])
        selected = ranked[0]
        oracle = min(candidates, key=lambda name: candidates[name]["test_spearman"])
        winners = fold_winners(x_val, y_val, folds=4)
        stability_unanimous = len(set(winners)) == 1

        top2 = ranked[:2]
        top3 = ranked[:3]
        negative = [name for name in ranked if candidates[name]["val_spearman"] < 0]
        if not negative:
            negative = [selected]
        qualities = np.array([-candidates[name]["val_spearman"] for name in negative], dtype=np.float64)
        weights = qualities / (qualities.sum() + 1e-12)

        methods = {
            "selected": candidates[selected]["test"],
            "top2_rank": rank_ensemble(candidates, top2, "test"),
            "top3_rank": rank_ensemble(candidates, top3, "test"),
            "neg_weighted_rank": rank_ensemble(candidates, negative, "test", weights),
            "stability_gated": rank_ensemble(candidates, top3, "test")
            if stability_unanimous
            else candidates[selected]["test"],
            "oracle": candidates[oracle]["test"],
        }
        row = {
            "case": case,
            "selected": selected,
            "oracle": oracle,
            "fold_winners": winners,
            "stability_unanimous": stability_unanimous,
            "top2": top2,
            "top3": top3,
            "negative_pool": negative,
        }
        for name, score in methods.items():
            mse, reduction = top_reduction(score, y_test, 0.25)
            row[f"{name}_spearman"] = metric(score, y_test)
            row[f"{name}_top25_mse"] = mse
            row[f"{name}_top25_reduction"] = reduction
        print(row)


if __name__ == "__main__":
    main()
