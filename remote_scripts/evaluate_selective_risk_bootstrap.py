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
    parser.add_argument("--boots", type=int, default=500)
    parser.add_argument("--frac", type=float, default=0.25)
    args = parser.parse_args()

    for case_id, case in enumerate(args.cases):
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
            "case": case,
            "hard_candidate": hard,
            "oracle_candidate": oracle,
            "ranked_candidates": ranked,
        }
        for i, (name, score) in enumerate(methods.items()):
            mse, reduction = top_reduction(score, y_test, args.frac)
            lo, hi, boot_mean = bootstrap_ci(
                score, y_test, args.frac, args.boots, seed=20260515 + 1000 * case_id + i
            )
            row[f"{name}_top25_mse"] = mse
            row[f"{name}_top25_reduction"] = reduction
            row[f"{name}_ci95"] = [lo, hi]
            row[f"{name}_boot_mean"] = boot_mean
        print(row)


if __name__ == "__main__":
    main()
