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


def forward_residual(exp, batch_x, batch_y, batch_x_mark, batch_y_mark):
    batch_x = batch_x.float().to(exp.device)
    batch_y = batch_y.float().to(exp.device)
    batch_x_mark = batch_x_mark.float().to(exp.device)
    batch_y_mark = batch_y_mark.float().to(exp.device)
    dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len :, :]).float()
    dec_inp = torch.cat([batch_y[:, : exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)
    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    f_dim = -1 if exp.args.features == "MS" else 0
    outputs = outputs[:, -exp.args.pred_len :, f_dim:]
    truth = batch_y[:, -exp.args.pred_len :, f_dim:]
    return ((outputs - truth) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()


def collect_test_corruption(exp, parr, noise_std=0.20, tail_frac=0.50, seed=20260515):
    _, loader = exp._get_data(flag="test")
    comps, clean_residuals, corrupt_residuals = [], [], []
    gen = torch.Generator(device="cpu").manual_seed(seed)
    exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            x_cpu = batch_x.float()
            patches, _ = parr._patchify(x_cpu)
            comp = torch.stack(
                [
                    parr._spectral_entropy(patches).mean(dim=1),
                    parr._period_drift(patches).mean(dim=1),
                    parr._smooth_residual(patches).mean(dim=1),
                    parr._channel_profile_drift(patches).mean(dim=1),
                ],
                dim=1,
            ).cpu().numpy()
            scale = x_cpu.std(dim=1, keepdim=True).clamp_min(1e-6)
            noise = torch.randn(x_cpu.shape, generator=gen) * scale * float(noise_std)
            mask = torch.zeros_like(x_cpu)
            start = int(round((1.0 - float(tail_frac)) * x_cpu.shape[1]))
            mask[:, start:, :] = 1.0
            x_corrupt = x_cpu + noise * mask

            clean = forward_residual(exp, x_cpu, batch_y, batch_x_mark, batch_y_mark)
            corrupt = forward_residual(exp, x_corrupt, batch_y, batch_x_mark, batch_y_mark)
            comps.append(comp)
            clean_residuals.append(clean)
            corrupt_residuals.append(corrupt)
    return np.concatenate(comps), np.concatenate(clean_residuals), np.concatenate(corrupt_residuals)


def top_mean(score, values, frac=0.25):
    order = np.argsort(score)
    k = max(1, int(round(frac * len(order))))
    return float(values[order[-k:]].mean())


def bottom_mean(score, values, frac=0.25):
    order = np.argsort(score)
    k = max(1, int(round(frac * len(order))))
    return float(values[order[:k]].mean())


def build_scores(candidates, y_test):
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
    }, selected, oracle


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
    parser.add_argument("--noise-std", type=float, default=0.20)
    parser.add_argument("--tail-frac", type=float, default=0.50)
    args = parser.parse_args()

    rows = []
    for case_id, case in enumerate(args.cases):
        cfg = base_args(case)
        ckpts = sorted(glob.glob(cfg.ckpt_glob))
        if not ckpts:
            print({"case": case, "error": "checkpoint_not_found", "glob": cfg.ckpt_glob}, flush=True)
            continue
        exp = Exp_Long_Term_Forecast(cfg)
        exp.model.load_state_dict(torch.load(ckpts[-1], map_location=exp.device))
        parr = PARRPreprocessor(cfg).eval()
        x_val, y_val, _ = collect_full(exp, parr, "val")
        x_test, y_clean, y_corrupt = collect_test_corruption(
            exp, parr, noise_std=args.noise_std, tail_frac=args.tail_frac, seed=20260515 + case_id
        )
        candidates = candidate_scores(x_val, y_val, x_test)
        for item in candidates.values():
            item["test_spearman"] = metric(item["test"], y_clean)
        methods, selected, oracle = build_scores(candidates, y_clean)
        delta = y_corrupt - y_clean
        row = {
            "case": case,
            "selected": selected,
            "oracle": oracle,
            "clean_mse": float(y_clean.mean()),
            "corrupt_mse": float(y_corrupt.mean()),
            "delta_mse": float(delta.mean()),
        }
        for name, score in methods.items():
            top_clean = top_mean(score, y_clean)
            top_corrupt = top_mean(score, y_corrupt)
            top_delta = top_mean(score, delta)
            low_delta = bottom_mean(score, delta)
            row[f"{name}_top25_clean_reduction"] = float(1.0 - top_clean / y_clean.mean())
            row[f"{name}_top25_corrupt_reduction"] = float(1.0 - top_corrupt / y_corrupt.mean())
            row[f"{name}_top25_delta_reduction"] = float(1.0 - top_delta / (delta.mean() + 1e-12))
            row[f"{name}_low25_delta"] = low_delta
            row[f"{name}_top25_delta"] = top_delta
        rows.append(row)
        print(row, flush=True)

    summary = {"rows": len(rows)}
    for method in ["hard", "top3_rank", "full_neg_weighted", "oracle"]:
        for key in ["top25_corrupt_reduction", "top25_delta_reduction"]:
            vals = [row[f"{method}_{key}"] for row in rows]
            summary[f"{method}_{key}_mean"] = float(np.mean(vals)) if vals else None
            summary[f"{method}_{key}_min"] = float(np.min(vals)) if vals else None
    print({"summary": summary}, flush=True)


if __name__ == "__main__":
    main()
