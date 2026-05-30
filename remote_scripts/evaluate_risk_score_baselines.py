import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "parr_icdm"))

from evaluate_calibrated_conformal import collect_full, fit_scores_for_val_test
from evaluate_val_calibrated_parr import base_args, sigmoid
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from layers.PARR import PARRPreprocessor


def standardize(x_val, x_test):
    mu = x_val.mean(axis=0)
    sigma = x_val.std(axis=0) + 1e-8
    return (x_val - mu) / sigma, (x_test - mu) / sigma


def metric(score, y):
    return float(spearmanr(score, y).statistic)


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
    args = parser.parse_args()
    names = ["spectral", "period", "smooth", "channel"]

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
        z_val, z_test = standardize(x_val, x_test)

        scores = fit_scores_for_val_test(x_val, y_val, x_test)
        for item in scores.values():
            item["test_spearman"] = metric(item["test"], y_test)

        fixed_beta = np.array([-1.0, 1.0, -1.0, -1.0])
        fixed_val = sigmoid(-(x_val @ fixed_beta))
        fixed_score = sigmoid(-(x_test @ fixed_beta))

        candidates = {
            "fixed_etth1": {"val": fixed_val, "test": fixed_score},
            "sign": {"val": scores["sign"]["val"], "test": scores["sign"]["test"]},
            "ridge": {"val": scores["ridge"]["val"], "test": scores["ridge"]["test"]},
        }
        single = {}
        for i, name in enumerate(names):
            corr_val = spearmanr(z_val[:, i], y_val).statistic
            sign = 1.0 if np.isnan(corr_val) or corr_val <= 0 else -1.0
            # high score should mean low risk.
            val_score = sign * z_val[:, i]
            test_score = sign * z_test[:, i]
            single[name] = metric(test_score, y_test)
            candidates[f"single_{name}"] = {"val": val_score, "test": test_score}

        for item in candidates.values():
            item["val_spearman"] = metric(item["val"], y_val)
            item["test_spearman"] = metric(item["test"], y_test)
        selected = min(candidates, key=lambda name: candidates[name]["val_spearman"])

        row = {
            "case": case,
            "fixed_etth1": metric(fixed_score, y_test),
            "sign_scorer": scores["sign"]["test_spearman"],
            "ridge_scorer": scores["ridge"]["test_spearman"],
            "selected_all": selected,
            "selected_all_val_spearman": candidates[selected]["val_spearman"],
            "selected_all_test_spearman": candidates[selected]["test_spearman"],
            "best_single_component": min(single, key=single.get),
            "best_single_spearman": min(single.values()),
            "single_components": single,
        }
        print(row)


if __name__ == "__main__":
    main()
