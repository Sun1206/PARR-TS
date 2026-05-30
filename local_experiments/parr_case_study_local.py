import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import risk_scoring_baseline_local as base
from rank_omega_diagnostics import fit_parr_pool, metric


COMPONENTS = ["spectral", "period", "smooth", "channel"]


def family_of(dataset):
    return "ETT" if dataset.startswith("ETT") else dataset


def parr_rank_score(x_val, r_val, x_test):
    candidates = fit_parr_pool(x_val, r_val, x_test)
    val_corr = {name: metric(scores[0], r_val) for name, scores in candidates.items()}
    selected = min(val_corr, key=val_corr.get)
    neg = [name for name, rho in val_corr.items() if rho < 0] or [selected]
    weights = np.array([max(-val_corr[name], 0.0) for name in neg], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(neg), dtype=np.float64)
    weights /= weights.sum()
    test_score = weights @ np.vstack([base.rank01(candidates[name][1]) for name in neg])
    return test_score, selected, ",".join(neg)


def evaluate_dataset(name, rel_path, seq_len, pred_len, patch_len, root, max_windows, frac, seed):
    arr = base.numeric_frame(Path(root) / rel_path)
    x, y = base.make_windows(arr, seq_len, pred_len, max_windows, seed=seed)
    residual = base.naive_residual(x, y)
    features = base.parr_components(x, patch_len)
    _, val_idx, test_idx = base.split_temporal(len(residual))

    x_val, r_val = features[val_idx], residual[val_idx]
    x_test, r_test = features[test_idx], residual[test_idx]
    score, selected, neg_pool = parr_rank_score(x_val, r_val, x_test)

    k = max(1, int(frac * len(score)))
    order = np.argsort(score)
    low_idx = order[-k:]
    high_idx = order[:k]
    overall = r_test.mean()

    row = {
        "dataset": name,
        "family": family_of(name),
        "seed": seed,
        "selected": selected,
        "neg_pool": neg_pool,
        "overall_residual": float(overall),
        "low_risk_residual": float(r_test[low_idx].mean()),
        "high_risk_residual": float(r_test[high_idx].mean()),
        "low_reduction": float(1.0 - r_test[low_idx].mean() / overall),
        "high_increase": float(r_test[high_idx].mean() / overall - 1.0),
        "risk_gap": float(r_test[high_idx].mean() / (r_test[low_idx].mean() + 1e-12)),
    }
    for i, comp in enumerate(COMPONENTS):
        row[f"{comp}_low"] = float(x_test[low_idx, i].mean())
        row[f"{comp}_high"] = float(x_test[high_idx, i].mean())
        row[f"{comp}_gap"] = float(x_test[high_idx, i].mean() - x_test[low_idx, i].mean())
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="local_experiments/results_multiseed")
    parser.add_argument("--max-windows", type=int, default=5000)
    parser.add_argument("--frac", type=float, default=0.25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 23])
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        for name, spec in base.DATASETS.items():
            path = Path(args.root) / spec[0]
            if not path.exists():
                print(f"skip {name}: missing {path}")
                continue
            print(f"seed {seed}: case study {name} ...")
            rows.append(evaluate_dataset(name, *spec, args.root, args.max_windows, args.frac, seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "parr_case_study.csv", index=False)

    summary = (
        df.groupby("family", as_index=False)
        .agg(
            runs=("dataset", "count"),
            low_reduction=("low_reduction", "mean"),
            high_increase=("high_increase", "mean"),
            risk_gap=("risk_gap", "mean"),
            spectral_gap=("spectral_gap", "mean"),
            period_gap=("period_gap", "mean"),
            smooth_gap=("smooth_gap", "mean"),
            channel_gap=("channel_gap", "mean"),
        )
        .sort_values("family")
    )
    all_row = pd.DataFrame(
        [
            {
                "family": "All",
                "runs": len(df),
                "low_reduction": df["low_reduction"].mean(),
                "high_increase": df["high_increase"].mean(),
                "risk_gap": df["risk_gap"].mean(),
                "spectral_gap": df["spectral_gap"].mean(),
                "period_gap": df["period_gap"].mean(),
                "smooth_gap": df["smooth_gap"].mean(),
                "channel_gap": df["channel_gap"].mean(),
            }
        ]
    )
    summary = pd.concat([summary, all_row], ignore_index=True)
    summary.to_csv(outdir / "parr_case_study_summary.csv", index=False)

    md = ["# PARR Case Study", ""]
    md.append("Top-25 PARR low-risk windows compared with bottom-25 high-risk windows.")
    md.append("")
    md.append(summary.to_markdown(index=False, floatfmt=".4f"))
    (outdir / "parr_case_study_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
