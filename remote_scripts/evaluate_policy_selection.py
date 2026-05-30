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


def temporal_folds(n, folds):
    edges = np.linspace(0, n, folds + 1, dtype=int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(folds)]


def candidate_scores(x_fit, y_fit, x_eval):
    mu = x_fit.mean(axis=0)
    sigma = x_fit.std(axis=0) + 1e-8
    z_fit = (x_fit - mu) / sigma
    z_eval = (x_eval - mu) / sigma
    scores = fit_scores_for_val_test(x_fit, y_fit, x_eval)
    candidates = {
        "fixed_etth1": {
            "fit": sigmoid(-(x_fit @ FIXED_BETA)),
            "eval": sigmoid(-(x_eval @ FIXED_BETA)),
        },
        "sign": {"fit": scores["sign"]["val"], "eval": scores["sign"]["test"]},
        "ridge": {"fit": scores["ridge"]["val"], "eval": scores["ridge"]["test"]},
    }
    for i, name in enumerate(COMPONENTS):
        corr_fit = spearman(z_fit[:, i], y_fit)
        direction = 1.0 if np.isinf(corr_fit) or corr_fit <= 0 else -1.0
        candidates[f"single_{name}"] = {
            "fit": direction * z_fit[:, i],
            "eval": direction * z_eval[:, i],
        }
    for item in candidates.values():
        item["fit_spearman"] = spearman(item["fit"], y_fit)
    return candidates


def rank_ensemble(candidates, names, split="eval", weights=None):
    if weights is None:
        weights = np.ones(len(names), dtype=np.float64) / len(names)
    ranks = np.vstack([rank01(candidates[name][split]) for name in names])
    return weights @ ranks


def materialize_policies(candidates):
    ranked = sorted(candidates, key=lambda name: candidates[name]["fit_spearman"])
    selected = ranked[0]
    top2 = ranked[:2]
    top3 = ranked[:3]
    negative = [name for name in ranked if candidates[name]["fit_spearman"] < 0]
    if not negative:
        negative = [selected]
    qualities = np.array([-candidates[name]["fit_spearman"] for name in negative], dtype=np.float64)
    weights = qualities / (qualities.sum() + 1e-12)

    policies = {
        "hard_selected": candidates[selected]["eval"],
        "top2_rank": rank_ensemble(candidates, top2),
        "top3_rank": rank_ensemble(candidates, top3),
        "negative_weighted_rank": rank_ensemble(candidates, negative, weights=weights),
    }
    for name in ranked:
        policies[f"candidate::{name}"] = candidates[name]["eval"]
    metadata = {"ranked": ranked, "selected": selected, "top2": top2, "top3": top3, "negative": negative}
    return policies, metadata


def cv_policy_scores(x_val, y_val, folds):
    policy_reductions = {}
    fold_best = []
    all_idx = np.arange(len(y_val))
    for hold_idx in temporal_folds(len(y_val), folds):
        fit_idx = np.setdiff1d(all_idx, hold_idx, assume_unique=True)
        candidates = candidate_scores(x_val[fit_idx], y_val[fit_idx], x_val[hold_idx])
        policies, _ = materialize_policies(candidates)
        fold_values = {}
        for name, score in policies.items():
            _, reduction = top_reduction(score, y_val[hold_idx])
            policy_reductions.setdefault(name, []).append(reduction)
            fold_values[name] = reduction
        fold_best.append(max(fold_values, key=fold_values.get))
    means = {name: float(np.mean(values)) for name, values in policy_reductions.items()}
    stds = {name: float(np.std(values)) for name, values in policy_reductions.items()}
    return means, stds, fold_best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "etth1_timemixer",
            "etth1_patchtst",
            "etth1_timexer",
            "etth2_timemixer",
            "etth2_patchtst",
            "etth2_timexer",
            "ettm1_timemixer",
            "ettm1_patchtst",
            "ettm1_timexer",
        ],
    )
    parser.add_argument("--folds", type=int, default=4)
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

        cv_mean, cv_std, fold_best = cv_policy_scores(x_val, y_val, args.folds)
        selected_policy = max(cv_mean, key=cv_mean.get)

        full_candidates = candidate_scores(x_val, y_val, x_test)
        full_policies, metadata = materialize_policies(full_candidates)
        for item in full_candidates.values():
            item["test_spearman"] = spearman(item["eval"], y_test)
        oracle_candidate = min(full_candidates, key=lambda name: full_candidates[name]["test_spearman"])
        oracle_score = full_candidates[oracle_candidate]["eval"]

        row = {
            "case": case,
            "selected_policy": selected_policy,
            "selected_policy_cv_top25_reduction": cv_mean[selected_policy],
            "selected_policy_cv_std": cv_std[selected_policy],
            "fold_best_policies": fold_best,
            "hard_selected_candidate": metadata["selected"],
            "ranked_candidates": metadata["ranked"],
            "oracle_candidate": oracle_candidate,
        }
        report_methods = {
            "policy_selected": full_policies[selected_policy],
            "hard_selected": full_policies["hard_selected"],
            "top2_rank": full_policies["top2_rank"],
            "top3_rank": full_policies["top3_rank"],
            "negative_weighted_rank": full_policies["negative_weighted_rank"],
            "oracle_candidate": oracle_score,
        }
        for name, score in report_methods.items():
            mse, reduction = top_reduction(score, y_test)
            row[f"{name}_spearman"] = spearman(score, y_test)
            row[f"{name}_top25_mse"] = mse
            row[f"{name}_top25_reduction"] = reduction
        print(row)


if __name__ == "__main__":
    main()
