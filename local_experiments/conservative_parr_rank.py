import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import rank_omega_diagnostics as diag
import risk_scoring_baseline_local as base


def lambda_value(mode, n_val, m_s, delta):
    if mode == "zero":
        return 0.0
    if mode == "fixed005":
        return 0.05
    if mode == "adaptive":
        return float(np.sqrt(np.log(max(m_s / delta, 1.0)) / (2.0 * max(n_val, 1))))
    raise ValueError(f"unknown lambda mode: {mode}")


def evaluate_dataset(name, rel_path, seq_len, pred_len, patch_len, root, max_windows, frac, seed, delta):
    arr = base.numeric_frame(Path(root) / rel_path)
    x, y = base.make_windows(arr, seq_len, pred_len, max_windows, seed=seed)
    residual = base.naive_residual(x, y)
    features = base.parr_components(x, patch_len)
    _, val_idx, test_idx = base.split_temporal(len(residual))

    x_val, r_val = features[val_idx], residual[val_idx]
    x_test, r_test = features[test_idx], residual[test_idx]
    candidates = diag.fit_parr_pool(x_val, r_val, x_test)
    val_corr = {name: diag.metric(scores[0], r_val) for name, scores in candidates.items()}
    selected = min(val_corr, key=val_corr.get)

    rows = []
    for mode in ["zero", "fixed005", "adaptive"]:
        lam = lambda_value(mode, len(r_val), len(candidates), delta)
        active_names = [cand for cand, rho in val_corr.items() if (-rho - lam) > 0]
        if not active_names and mode == "zero":
            active_names = [selected]

        if active_names:
            weights = np.array([max(-val_corr[cand] - lam, 0.0) for cand in active_names], dtype=np.float64)
            if weights.sum() <= 0:
                weights = np.ones(len(active_names), dtype=np.float64)
            weights /= weights.sum()
            ranks_test = np.vstack([base.rank01(candidates[cand][1]) for cand in active_names])
            score_test = weights @ ranks_test
            reduction = base.top_reduction(score_test, r_test, frac)
            spearman = diag.metric(score_test, r_test)
            omega_lam = float(sum(max(-val_corr[cand] - lam, 0.0) for cand in active_names))
            sign_rate, omega_block_min = diag.block_sign_rate(candidates, r_val, active_names)
        else:
            reduction = np.nan
            spearman = np.nan
            omega_lam = 0.0
            sign_rate = np.nan
            omega_block_min = np.nan

        rows.append(
            {
                "dataset": name,
                "family": diag.family_of(name),
                "seed": seed,
                "lambda_mode": mode,
                "lambda": lam,
                "active": bool(active_names),
                "n_val": len(r_val),
                "pool_size": len(active_names),
                "omega_lambda": omega_lam,
                "omega_block_min": omega_block_min,
                "block_sign_rate": sign_rate,
                "test_spearman": spearman,
                "top25_reduction": reduction,
                "selected_pool": ",".join(active_names),
            }
        )
    return rows


def summarize(df):
    active = df[df["active"]].copy()
    summary = (
        df.groupby("lambda_mode", as_index=False)
        .agg(
            cases=("dataset", "count"),
            active=("active", "sum"),
            lambda_mean=("lambda", "mean"),
        )
        .merge(
            active.groupby("lambda_mode", as_index=False).agg(
                pool_mean=("pool_size", "mean"),
                omega_lambda_mean=("omega_lambda", "mean"),
                block_sign_rate=("block_sign_rate", "mean"),
                top25_reduction_mean=("top25_reduction", "mean"),
                top25_reduction_min=("top25_reduction", "min"),
                active_negative=("top25_reduction", lambda s: int((s < 0).sum())),
            ),
            on="lambda_mode",
            how="left",
        )
    )
    order = {"zero": 0, "fixed005": 1, "adaptive": 2}
    return summary.sort_values("lambda_mode", key=lambda s: s.map(order)).reset_index(drop=True)


def summarize_family(df):
    active = df[df["active"]].copy()
    summary = (
        active.groupby(["lambda_mode", "family"], as_index=False)
        .agg(
            runs=("dataset", "count"),
            pool_mean=("pool_size", "mean"),
            omega_lambda_mean=("omega_lambda", "mean"),
            block_sign_rate=("block_sign_rate", "mean"),
            top25_reduction_mean=("top25_reduction", "mean"),
            top25_reduction_min=("top25_reduction", "min"),
        )
    )
    order = {"zero": 0, "fixed005": 1, "adaptive": 2}
    return summary.sort_values(["lambda_mode", "family"], key=lambda s: s.map(order).fillna(s)).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="local_experiments/results_multiseed")
    parser.add_argument("--max-windows", type=int, default=5000)
    parser.add_argument("--frac", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 23])
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        for name, spec in base.DATASETS.items():
            path = Path(args.root) / spec[0]
            if not path.exists():
                print(f"skip {name}: missing {path}")
                continue
            print(f"seed {seed}: conservative PARR-rank {name} ...")
            rows.extend(evaluate_dataset(name, *spec, args.root, args.max_windows, args.frac, seed, args.delta))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "conservative_parr_rank.csv", index=False)
    summary = summarize(df)
    summary.to_csv(outdir / "conservative_parr_rank_summary.csv", index=False)
    family_summary = summarize_family(df)
    family_summary.to_csv(outdir / "conservative_parr_rank_family_summary.csv", index=False)

    md = ["# Conservative PARR-rank", ""]
    md.append("Lambda modes: zero is the default point estimate, fixed005 uses lambda=0.05, and adaptive uses sqrt(log(M/delta)/(2 n_val)).")
    md.append("")
    md.append(summary.to_markdown(index=False, floatfmt=".4f"))
    md.append("")
    md.append("## Family summary")
    md.append("")
    md.append(family_summary.to_markdown(index=False, floatfmt=".4f"))
    (outdir / "conservative_parr_rank_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
