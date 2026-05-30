from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local_experiments"))

import risk_scoring_baseline_local as base  # noqa: E402
from rank_omega_diagnostics import fit_parr_pool, metric  # noqa: E402


def parr_score(x_val, r_val, x_test):
    candidates = fit_parr_pool(x_val, r_val, x_test)
    corr = {name: metric(scores[0], r_val) for name, scores in candidates.items()}
    selected = min(corr, key=corr.get)
    neg = [name for name, rho in corr.items() if rho < 0] or [selected]
    weights = np.array([max(-corr[name], 0.0) for name in neg], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(neg))
    weights /= weights.sum()
    return weights @ np.vstack([base.rank01(candidates[name][1]) for name in neg])


def zline(x):
    y = x.mean(axis=1)
    return (y - y.mean()) / (y.std() + 1e-8)


def main() -> None:
    arr = base.numeric_frame(ROOT / "data_cache" / "ETTm2.csv")
    x, y = base.make_windows(arr, 96, 96, 5000, seed=7)
    residual = base.naive_residual(x, y)
    features = base.parr_components(x, 16)
    _, val_idx, test_idx = base.split_temporal(len(residual))
    score = parr_score(features[val_idx], residual[val_idx], features[test_idx])
    test_x = x[test_idx]
    test_r = residual[test_idx]
    test_f = features[test_idx]
    k = max(1, len(score) // 4)
    low_pool = np.argsort(score)[-k:]
    high_pool = np.argsort(score)[:k]
    low = int(low_pool[np.argmin(test_r[low_pool])])
    high = int(high_pool[np.argmax(test_r[high_pool])])

    scaler = StandardScaler().fit(features[val_idx])
    bars = scaler.transform(test_f[[low, high]])
    comps = ["spectral", "period", "smooth", "channel"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.2,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(6.85, 1.55), gridspec_kw={"width_ratios": [1.55, 1.25, 0.7]})

    t = np.arange(test_x.shape[1])
    axes[0].plot(t, zline(test_x[low]), label="selected low-risk", linewidth=0.9)
    axes[0].plot(t, zline(test_x[high]), label="rejected high-risk", linewidth=0.9)
    axes[0].set_xlabel("Input time")
    axes[0].set_ylabel("Mean z-value")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(axis="y", linewidth=0.25, alpha=0.35)

    width = 0.34
    xpos = np.arange(len(comps))
    axes[1].bar(xpos - width / 2, bars[0], width, label="low-risk")
    axes[1].bar(xpos + width / 2, bars[1], width, label="high-risk")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(comps, rotation=20, ha="right")
    axes[1].set_ylabel("Calibrated z")
    axes[1].grid(axis="y", linewidth=0.25, alpha=0.35)

    axes[2].bar([0, 1], [test_r[low], test_r[high]], width=0.58, color=["#1f77b4", "#d62728"])
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["low", "high"])
    axes[2].set_ylabel("MSE")
    axes[2].grid(axis="y", linewidth=0.25, alpha=0.35)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)

    fig.tight_layout(pad=0.2, w_pad=0.75)
    out = Path(__file__).resolve().parent
    fig.savefig(out / "case_study_components.pdf")
    fig.savefig(out / "case_study_components.png", dpi=300)


if __name__ == "__main__":
    main()
