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


def rank01(score):
    if len(score) <= 1:
        return np.zeros_like(score, dtype=np.float64)
    return (rankdata(score, method="average") - 1.0) / (len(score) - 1.0)


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
        scores = fit_scores_for_val_test(x_val, y_val, x_test)

        mu = x_val.mean(axis=0)
        sigma = x_val.std(axis=0) + 1e-8
        z_val = (x_val - mu) / sigma
        z_test = (x_test - mu) / sigma
        fixed_beta = np.array([-1.0, 1.0, -1.0, -1.0])
        candidates = {
            "fixed_etth1": {"val": sigmoid(-(x_val @ fixed_beta)), "test": sigmoid(-(x_test @ fixed_beta))},
            "sign": {"val": scores["sign"]["val"], "test": scores["sign"]["test"]},
            "ridge": {"val": scores["ridge"]["val"], "test": scores["ridge"]["test"]},
        }
        for i, name in enumerate(["spectral", "period", "smooth", "channel"]):
            corr_val = spearmanr(z_val[:, i], y_val).statistic
            sign = 1.0 if np.isnan(corr_val) or corr_val <= 0 else -1.0
            candidates[f"single_{name}"] = {"val": sign * z_val[:, i], "test": sign * z_test[:, i]}

        for item in candidates.values():
            item["val_spearman"] = float(spearmanr(item["val"], y_val).statistic)
            item["test_spearman"] = float(spearmanr(item["test"], y_test).statistic)
        selected = min(candidates, key=lambda name: candidates[name]["val_spearman"])
        oracle = min(candidates, key=lambda name: candidates[name]["test_spearman"])
        ranked = sorted(candidates, key=lambda name: candidates[name]["val_spearman"])
        negative = [name for name in ranked if candidates[name]["val_spearman"] < 0]
        if not negative:
            negative = [selected]
        qualities = np.array([-candidates[name]["val_spearman"] for name in negative], dtype=np.float64)
        weights = qualities / (qualities.sum() + 1e-12)
        parr_rank_score = rank_ensemble(candidates, negative, weights)
        score = candidates[selected]["test"]
        order = np.argsort(score)
        parr_rank_order = np.argsort(parr_rank_score)
        overall = float(y_test.mean())
        row = {
            "case": case,
            "selected": selected,
            "parr_rank_pool": negative,
            "val_spearman": candidates[selected]["val_spearman"],
            "test_spearman": candidates[selected]["test_spearman"],
            "parr_rank_test_spearman": float(spearmanr(parr_rank_score, y_test).statistic),
            "test_oracle": oracle,
            "test_oracle_spearman": candidates[oracle]["test_spearman"],
            "overall_mse": overall,
            "low25_mse": float(y_test[order[: max(1, int(0.25 * len(order)))]].mean()),
        }
        for frac in [0.1, 0.25, 0.5]:
            k = max(1, int(frac * len(order)))
            mse = float(y_test[order[-k:]].mean())
            row[f"top{int(frac * 100)}_mse"] = mse
            row[f"top{int(frac * 100)}_reduction"] = float(1.0 - mse / overall)
            parr_rank_mse = float(y_test[parr_rank_order[-k:]].mean())
            row[f"parr_rank_top{int(frac * 100)}_mse"] = parr_rank_mse
            row[f"parr_rank_top{int(frac * 100)}_reduction"] = float(1.0 - parr_rank_mse / overall)
            oracle_order = np.argsort(candidates[oracle]["test"])
            oracle_mse = float(y_test[oracle_order[-k:]].mean())
            row[f"oracle_top{int(frac * 100)}_mse"] = oracle_mse
            row[f"oracle_top{int(frac * 100)}_reduction"] = float(1.0 - oracle_mse / overall)
        print(row)


if __name__ == "__main__":
    main()
