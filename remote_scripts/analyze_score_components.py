import argparse
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.getcwd())

from data_provider.data_factory import data_provider
from layers.PARR import PARRPreprocessor


def corr(x, y):
    return {
        "spearman": float(spearmanr(x, y).statistic),
        "pearson": float(pearsonr(x, y).statistic),
    }


def make_args(batch_size):
    return SimpleNamespace(
        task_name="long_term_forecast",
        root_path="./dataset/ETT-small/",
        data_path="ETTh1.csv",
        data="ETTh1",
        features="M",
        target="OT",
        freq="h",
        embed="timeF",
        seq_len=96,
        label_len=0,
        pred_len=96,
        batch_size=batch_size,
        num_workers=0,
        seasonal_patterns="Monthly",
        parr_patch_len=16,
        parr_alpha_s=1.0,
        parr_alpha_d=0.5,
        parr_alpha_e=1.0,
        parr_alpha_g=0.5,
        parr_min_keep=1e-4,
        parr_dropout=0.0,
        parr_replace_strength=0.0,
    )


def collect_components(batch_size):
    args = make_args(batch_size)
    _, loader = data_provider(args, "test")
    scorer = PARRPreprocessor(args).eval()
    rows = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            x = batch_x.float()
            patches, _ = scorer._patchify(x)
            spectral = scorer._spectral_entropy(patches).mean(dim=1).cpu().numpy()
            period = scorer._period_drift(patches).mean(dim=1).cpu().numpy()
            residual = scorer._smooth_residual(patches).mean(dim=1).cpu().numpy()
            channel = scorer._channel_profile_drift(patches).mean(dim=1).cpu().numpy()
            rows.append(np.stack([spectral, period, residual, channel], axis=1))
    return np.concatenate(rows, axis=0)


def find_result_folder(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No result folder matches {pattern}")
    return matches[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_glob",
        default="results/long_term_forecast_first_real_timemixer_etth1_96_TimeMixer_ETTh1_*first_real_base_0",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    folder = find_result_folder(args.result_glob)
    pred = np.load(os.path.join(folder, "pred.npy"))
    true = np.load(os.path.join(folder, "true.npy"))
    sample_residual = ((pred - true) ** 2).mean(axis=(1, 2))
    components = collect_components(args.batch_size)
    n = min(len(sample_residual), len(components))
    sample_residual = sample_residual[:n]
    components = components[:n]

    names = ["spectral_entropy", "period_drift", "smooth_residual", "channel_profile_drift"]
    print("folder", folder)
    print("n", n, "residual_mean", float(sample_residual.mean()))
    for idx, name in enumerate(names):
        vals = components[:, idx]
        out = corr(vals, sample_residual)
        print(name, "mean", float(vals.mean()), "std", float(vals.std()), out)

    # Grid a few interpretable risk scores. Positive risk-residual correlation is useful;
    # predictability can then be defined as exp(-risk).
    grids = {
        "all_original": np.array([1.0, 0.5, 1.0, 0.5]),
        "spectral_only": np.array([1.0, 0.0, 0.0, 0.0]),
        "period_only": np.array([0.0, 1.0, 0.0, 0.0]),
        "smooth_only": np.array([0.0, 0.0, 1.0, 0.0]),
        "channel_only": np.array([0.0, 0.0, 0.0, 1.0]),
        "no_spectral": np.array([0.0, 0.5, 1.0, 0.5]),
        "no_period": np.array([1.0, 0.0, 1.0, 0.5]),
        "no_smooth": np.array([1.0, 0.5, 0.0, 0.5]),
        "no_channel": np.array([1.0, 0.5, 1.0, 0.0]),
    }
    for name, weights in grids.items():
        risk = components @ weights
        pred_score = np.exp(-risk)
        print(
            "grid",
            name,
            "risk_corr",
            corr(risk, sample_residual),
            "predictability_corr",
            corr(pred_score, sample_residual),
        )


if __name__ == "__main__":
    main()
