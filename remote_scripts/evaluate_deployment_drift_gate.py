import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "parr_icdm"))

from evaluate_calibrated_conformal import collect_full
from evaluate_score_ensembles import candidate_scores, metric, rank_ensemble
from evaluate_val_calibrated_parr import base_args
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from layers.PARR import PARRPreprocessor


DEFAULT_CASES = [
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
]


def top_reduction(score, residual, frac=0.25):
    order = np.argsort(score)
    k = max(1, int(frac * len(order)))
    mse = float(residual[order[-k:]].mean())
    overall = float(residual.mean())
    return mse, float(1.0 - mse / overall)


def mean_shift_std(source, target):
    return float(abs(np.mean(target) - np.mean(source)) / (np.std(source) + 1e-8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--shift-threshold", type=float, default=0.60)
    parser.add_argument("--min-val-windows", type=int, default=200)
    parser.add_argument("--frac", type=float, default=0.25)
    args = parser.parse_args()

    rows = []
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

        ranked = sorted(candidates, key=lambda name: candidates[name]["val_spearman"])
        selected = ranked[0]
        negative = [name for name in ranked if candidates[name]["val_spearman"] < 0] or [selected]
        qualities = np.array([-candidates[name]["val_spearman"] for name in negative], dtype=np.float64)
        weights = qualities / (qualities.sum() + 1e-12)
        full_val = rank_ensemble(candidates, negative, "val", weights)
        full_test = rank_ensemble(candidates, negative, "test", weights)

        selected_val = candidates[selected]["val"]
        selected_test = candidates[selected]["test"]
        selected_shift = mean_shift_std(selected_val, selected_test)
        full_shift = mean_shift_std(full_val, full_test)
        shift_reliable = selected_shift <= args.shift_threshold
        sample_reliable = len(y_val) >= args.min_val_windows
        deployable = bool(shift_reliable and sample_reliable)
        gate_reasons = []
        if not shift_reliable:
            gate_reasons.append("score_shift")
        if not sample_reliable:
            gate_reasons.append("insufficient_val_windows")
        if not gate_reasons:
            gate_reasons.append("pass")

        selected_mse, selected_reduction = top_reduction(selected_test, y_test, args.frac)
        full_mse, full_reduction = top_reduction(full_test, y_test, args.frac)
        row = {
            "case": case,
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "selected": selected,
            "val_spearman": candidates[selected]["val_spearman"],
            "test_spearman": candidates[selected]["test_spearman"],
            "selected_score_ks": float(ks_2samp(selected_val, selected_test).statistic),
            "selected_score_wasserstein": float(wasserstein_distance(selected_val, selected_test)),
            "selected_score_mean_shift_std": selected_shift,
            "full_rank_mean_shift_std": full_shift,
            "min_val_windows": args.min_val_windows,
            "shift_reliable": shift_reliable,
            "sample_reliable": sample_reliable,
            "deployable": deployable,
            "gate": "apply" if deployable else "abstain",
            "gate_reasons": gate_reasons,
            "selected_top25_mse": selected_mse,
            "selected_top25_reduction": selected_reduction,
            "full_top25_mse": full_mse,
            "full_top25_reduction": full_reduction,
        }
        rows.append(row)
        print(row, flush=True)

    summary = {
        "rows": len(rows),
        "shift_threshold": args.shift_threshold,
        "min_val_windows": args.min_val_windows,
        "applied": sum(row["deployable"] for row in rows),
        "abstained": sum(not row["deployable"] for row in rows),
        "negative_full_cases": [row["case"] for row in rows if row["full_top25_reduction"] < 0],
        "abstained_cases": [row["case"] for row in rows if not row["deployable"]],
    }
    applied = [row for row in rows if row["deployable"]]
    if applied:
        summary["applied_full_reduction_mean"] = float(np.mean([row["full_top25_reduction"] for row in applied]))
        summary["applied_full_reduction_min"] = float(np.min([row["full_top25_reduction"] for row in applied]))
        summary["applied_full_positive"] = int(sum(row["full_top25_reduction"] > 0 for row in applied))
    print({"summary": summary}, flush=True)


if __name__ == "__main__":
    main()
