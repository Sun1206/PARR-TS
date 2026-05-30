import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "parr_icdm"))

from evaluate_calibrated_conformal import collect_full
from evaluate_val_calibrated_parr import base_args, sigmoid
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from layers.PARR import PARRPreprocessor


COMPONENTS = ["spectral", "period", "smooth", "channel"]
FIXED_BETA = np.array([-1.0, 1.0, -1.0, -1.0])


def metric(score, y):
    value = spearmanr(score, y).statistic
    return float(value) if not np.isnan(value) else float("inf")


def build_scores(x_train, y_train, x_eval):
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0) + 1e-8
    z_train = (x_train - mu) / sigma
    z_eval = (x_eval - mu) / sigma

    out = {"fixed_etth1": sigmoid(-(x_eval @ FIXED_BETA))}

    signs = []
    for i in range(z_train.shape[1]):
        s = metric(z_train[:, i], y_train)
        signs.append(0.0 if abs(s) < 0.03 or np.isinf(s) else float(np.sign(s)))
    signs = np.asarray(signs)
    out["sign"] = sigmoid(-(z_eval @ signs))

    x1_train = np.c_[np.ones(len(z_train)), z_train]
    x1_eval = np.c_[np.ones(len(z_eval)), z_eval]
    ridge = 1e-3 * np.eye(x1_train.shape[1])
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x1_train.T @ x1_train + ridge, x1_train.T @ y_train)
    out["ridge"] = -(x1_eval @ beta)

    for i, name in enumerate(COMPONENTS):
        s = metric(z_train[:, i], y_train)
        direction = 1.0 if np.isinf(s) or s <= 0 else -1.0
        out[f"single_{name}"] = direction * z_eval[:, i]
    return out


def temporal_folds(n, folds):
    edges = np.linspace(0, n, folds + 1, dtype=int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(folds)]


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

        full_val_scores = build_scores(x_val, y_val, x_val)
        full_test_scores = build_scores(x_val, y_val, x_test)
        full_val = {name: metric(score, y_val) for name, score in full_val_scores.items()}
        test = {name: metric(score, y_test) for name, score in full_test_scores.items()}
        full_selected = min(full_val, key=full_val.get)
        test_oracle = min(test, key=test.get)

        fold_metrics = {name: [] for name in full_val_scores}
        fold_winners = []
        all_idx = np.arange(len(y_val))
        for hold_idx in temporal_folds(len(y_val), args.folds):
            train_idx = np.setdiff1d(all_idx, hold_idx, assume_unique=True)
            hold_scores = build_scores(x_val[train_idx], y_val[train_idx], x_val[hold_idx])
            hold_metric = {name: metric(score, y_val[hold_idx]) for name, score in hold_scores.items()}
            for name, value in hold_metric.items():
                fold_metrics[name].append(value)
            fold_winners.append(min(hold_metric, key=hold_metric.get))

        cv_mean = {name: float(np.mean(values)) for name, values in fold_metrics.items()}
        cv_std = {name: float(np.std(values)) for name, values in fold_metrics.items()}
        cv_selected = min(cv_mean, key=cv_mean.get)

        row = {
            "case": case,
            "full_selected": full_selected,
            "full_selected_val": full_val[full_selected],
            "full_selected_test": test[full_selected],
            "cv_selected": cv_selected,
            "cv_selected_mean": cv_mean[cv_selected],
            "cv_selected_std": cv_std[cv_selected],
            "cv_selected_test": test[cv_selected],
            "test_oracle": test_oracle,
            "test_oracle_spearman": test[test_oracle],
            "fold_winners": fold_winners,
            "winner_counts": dict(Counter(fold_winners)),
        }
        print(row)


if __name__ == "__main__":
    main()
