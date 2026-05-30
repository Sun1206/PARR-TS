import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "parr_icdm"))

from evaluate_calibrated_conformal import collect_full
from evaluate_score_ensembles import candidate_scores, metric, rank_ensemble
from evaluate_val_calibrated_parr import base_args
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from layers.PARR import PARRPreprocessor


def selective_mse(score, y, coverage):
    order = np.argsort(score)
    k = max(1, int(round(float(coverage) * len(order))))
    return float(y[order[-k:]].mean())


def build_methods(candidates, y_test):
    ranked = sorted(candidates, key=lambda name: candidates[name]["val_spearman"])
    selected = ranked[0]
    oracle = min(candidates, key=lambda name: candidates[name]["test_spearman"])
    negative = [name for name in ranked if candidates[name]["val_spearman"] < 0]
    if not negative:
        negative = [selected]
    qualities = np.array([-candidates[name]["val_spearman"] for name in negative], dtype=np.float64)
    weights = qualities / (qualities.sum() + 1e-12)
    return {
        "hard": candidates[selected]["test"],
        "top3_rank": rank_ensemble(candidates, ranked[:3], "test"),
        "full_neg_weighted": rank_ensemble(candidates, negative, "test", weights),
        "oracle": candidates[oracle]["test"],
    }, selected, oracle, ranked, negative


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
        ],
    )
    parser.add_argument("--coverages", nargs="+", type=float, default=[0.10, 0.25, 0.50, 0.75, 1.00])
    args = parser.parse_args()

    all_rows = []
    for case in args.cases:
        cfg = base_args(case)
        ckpts = sorted(glob.glob(cfg.ckpt_glob))
        if not ckpts:
            print({"case": case, "error": "checkpoint_not_found", "glob": cfg.ckpt_glob}, flush=True)
            continue
        exp = Exp_Long_Term_Forecast(cfg)
        exp.model.load_state_dict(torch.load(ckpts[-1], map_location=exp.device))
        parr = PARRPreprocessor(cfg).eval()
        x_val, y_val, _ = collect_full(exp, parr, "val")
        x_test, y_test, _ = collect_full(exp, parr, "test")
        candidates = candidate_scores(x_val, y_val, x_test)
        for item in candidates.values():
            item["test_spearman"] = metric(item["test"], y_test)
        methods, selected, oracle, ranked, negative = build_methods(candidates, y_test)
        overall = float(y_test.mean())

        row = {
            "case": case,
            "overall_mse": overall,
            "selected": selected,
            "oracle": oracle,
            "ranked": ranked,
            "negative_pool": negative,
        }
        for method, score in methods.items():
            ratios = []
            for cov in args.coverages:
                mse = selective_mse(score, y_test, cov)
                row[f"{method}_cov{int(cov * 100):02d}_mse"] = mse
                row[f"{method}_cov{int(cov * 100):02d}_ratio"] = mse / overall
                ratios.append(mse / overall)
            row[f"{method}_mean_ratio_excl100"] = float(np.mean(ratios[:-1]))
            row[f"{method}_auc_reduction_excl100"] = float(1.0 - np.mean(ratios[:-1]))
        all_rows.append(row)
        print(row, flush=True)

    summary = {"rows": len(all_rows)}
    for method in ["hard", "top3_rank", "full_neg_weighted", "oracle"]:
        values = [row[f"{method}_auc_reduction_excl100"] for row in all_rows]
        summary[f"{method}_auc_reduction_mean"] = float(np.mean(values)) if values else None
        summary[f"{method}_auc_reduction_min"] = float(np.min(values)) if values else None
    print({"summary": summary}, flush=True)


if __name__ == "__main__":
    main()
