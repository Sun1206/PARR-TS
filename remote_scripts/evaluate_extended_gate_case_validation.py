"""Recompute the extended PARR-TS deployment gate from frozen checkpoints.

The sign-stability statistic is intentionally validation-only. Test residuals
are collected only after all gate quantities have been fixed, and are used only
for the final audit column.
"""

import argparse
import csv
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
    "electricity_timemixer",
    "electricity_patchtst",
    "electricity_timexer",
    "electricity_itransformer",
    "weather_timemixer",
    "weather_patchtst",
    "weather_timexer",
    "weather_itransformer",
    "exchange_timemixer",
    "exchange_patchtst",
    "exchange_timexer",
    "exchange_itransformer",
    "illness_timemixer",
    "illness_patchtst",
    "illness_timexer",
    "illness_itransformer",
]


def top_reduction(score, residual, frac=0.25):
    order = np.argsort(score)
    k = max(1, int(frac * len(order)))
    selected = float(residual[order[-k:]].mean())
    overall = float(residual.mean())
    return float(1.0 - selected / overall)


def mean_shift_std(source, target):
    return float(abs(np.mean(target) - np.mean(source)) / (np.std(source) + 1e-8))


def block_sign_stability(score, residual, blocks):
    signs = []
    for idx in np.array_split(np.arange(len(residual)), blocks):
        if len(idx) < 4:
            continue
        signs.append(metric(score[idx], residual[idx]) < 0)
    if not signs:
        return 0.0
    return float(np.mean(signs))


def evaluate_case(case, args):
    cfg = base_args(case)
    ckpts = sorted(glob.glob(cfg.ckpt_glob))
    if not ckpts:
        return {
            "case": case,
            "error": f"checkpoint_not_found: {cfg.ckpt_glob}",
        }

    exp = Exp_Long_Term_Forecast(cfg)
    exp.model.load_state_dict(torch.load(ckpts[-1], map_location=exp.device))
    parr = PARRPreprocessor(cfg).eval()
    x_val, y_val, _ = collect_full(exp, parr, "val")
    x_test, y_test, _ = collect_full(exp, parr, "test")

    candidates = candidate_scores(x_val, y_val, x_test)
    ranked = sorted(candidates, key=lambda name: candidates[name]["val_spearman"])
    selected = ranked[0]
    negative = [name for name in ranked if candidates[name]["val_spearman"] < 0] or [selected]
    qualities = np.array([-candidates[name]["val_spearman"] for name in negative], dtype=np.float64)
    weights = qualities / (qualities.sum() + 1e-12)
    parr_val = rank_ensemble(candidates, negative, "val", weights)
    parr_test = rank_ensemble(candidates, negative, "test", weights)

    selected_val = candidates[selected]["val"]
    selected_test = candidates[selected]["test"]
    delta_score = mean_shift_std(selected_val, selected_test)
    b_sign = block_sign_stability(parr_val, y_val, args.blocks)
    gate_before_sign = len(y_val) >= args.min_val_windows and delta_score <= args.shift_threshold
    final_gate = gate_before_sign and b_sign >= args.sign_threshold

    return {
        "case": case,
        "n_val": int(len(y_val)),
        "Delta_score": delta_score,
        "B_sign": b_sign,
        "gate before sign": "apply" if gate_before_sign else "abstain",
        "final gate": "apply" if final_gate else "abstain",
        "PARR Top-25": top_reduction(parr_test, y_test, args.frac),
        "selected": selected,
        "negative_pool": ",".join(negative),
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--out", default="logs/parr_icdm/extended_gate_case_validation.csv")
    parser.add_argument("--shift-threshold", type=float, default=0.60)
    parser.add_argument("--sign-threshold", type=float, default=2.0 / 3.0)
    parser.add_argument("--min-val-windows", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--frac", type=float, default=0.25)
    args = parser.parse_args()

    rows = []
    for case in args.cases:
        row = evaluate_case(case, args)
        rows.append(row)
        print(row, flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = [
        "case",
        "n_val",
        "Delta_score",
        "B_sign",
        "gate before sign",
        "final gate",
        "PARR Top-25",
        "selected",
        "negative_pool",
        "error",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
