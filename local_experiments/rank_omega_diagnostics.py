import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import risk_scoring_baseline_local as base


def metric(score, residual):
    if len(score) < 2 or np.std(score) <= 1e-12 or np.std(residual) <= 1e-12:
        return 0.0
    value = spearmanr(score, residual).statistic
    return float(value) if not np.isnan(value) else 0.0


def fit_parr_pool(x_val, r_val, x_test):
    scaler = StandardScaler().fit(x_val)
    zv = scaler.transform(x_val)
    zt = scaler.transform(x_test)
    candidates = {}

    fixed_beta = np.array([-1.0, 1.0, -1.0, -1.0])
    candidates["fixed_formula"] = (-(x_val @ fixed_beta), -(x_test @ fixed_beta))

    directions = []
    for i, name in enumerate(["spectral", "period", "smooth", "channel"]):
        corr = metric(zv[:, i], r_val)
        direction = 1.0 if corr <= 0 else -1.0
        directions.append(direction)
        candidates[f"single_{name}"] = (direction * zv[:, i], direction * zt[:, i])

    directions = np.array(directions)
    candidates["sign_linear"] = (zv @ directions, zt @ directions)

    ridge = Ridge(alpha=1.0).fit(zv, r_val)
    candidates["ridge_risk"] = (-ridge.predict(zv), -ridge.predict(zt))
    return candidates


def top_set(score, frac, high=True):
    k = max(1, int(frac * len(score)))
    order = np.argsort(score)
    return set(order[-k:] if high else order[:k]), k


def family_of(dataset):
    return "ETT" if dataset.startswith("ETT") else dataset


def block_sign_rate(candidates, r_val, neg_names, blocks=4):
    if not neg_names:
        return np.nan, np.nan
    indices = np.array_split(np.arange(len(r_val)), blocks)
    signs = []
    omegas = []
    for idx in indices:
        if len(idx) < 4:
            continue
        block_corrs = {name: metric(candidates[name][0][idx], r_val[idx]) for name in neg_names}
        signs.extend([rho < 0 for rho in block_corrs.values()])
        omegas.append(sum(-rho for rho in block_corrs.values() if rho < 0))
    if not signs:
        return np.nan, np.nan
    return float(np.mean(signs)), float(np.min(omegas)) if omegas else np.nan


def evaluate_dataset(name, rel_path, seq_len, pred_len, patch_len, root, max_windows, frac, seed):
    arr = base.numeric_frame(Path(root) / rel_path)
    x, y = base.make_windows(arr, seq_len, pred_len, max_windows, seed=seed)
    residual = base.naive_residual(x, y)
    features = base.parr_components(x, patch_len)
    _, val_idx, test_idx = base.split_temporal(len(residual))

    x_val, r_val = features[val_idx], residual[val_idx]
    x_test, r_test = features[test_idx], residual[test_idx]
    candidates = fit_parr_pool(x_val, r_val, x_test)
    val_corr = {name: metric(scores[0], r_val) for name, scores in candidates.items()}
    test_corr = {name: metric(scores[1], r_test) for name, scores in candidates.items()}

    selected = min(val_corr, key=val_corr.get)
    neg_names = [name for name, rho in val_corr.items() if rho < 0] or [selected]
    weights = np.array([max(-val_corr[name], 0.0) for name in neg_names], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(neg_names), dtype=np.float64)
    weights /= weights.sum()

    ranks_val = np.vstack([base.rank01(candidates[name][0]) for name in neg_names])
    ranks_test = np.vstack([base.rank01(candidates[name][1]) for name in neg_names])
    parr_val = weights @ ranks_val
    parr_test = weights @ ranks_test

    selected_set, k = top_set(parr_test, frac, high=True)
    oracle_set, _ = top_set(-r_test, frac, high=True)
    overlap = len(selected_set & oracle_set) / k
    eps_hat = 2.0 * frac * (1.0 - overlap)

    reduction = base.top_reduction(parr_test, r_test, frac)
    oracle_reduction = base.top_reduction(-r_test, r_test, frac)
    sign_rate, omega_block_min = block_sign_rate(candidates, r_val, neg_names)

    return {
        "dataset": name,
        "family": family_of(name),
        "seed": seed,
        "n_val": len(r_val),
        "n_test": len(r_test),
        "neg_count": len(neg_names),
        "omega": sum(-val_corr[name] for name in neg_names if val_corr[name] < 0),
        "omega_block_min": omega_block_min,
        "block_sign_rate": sign_rate,
        "val_spearman": metric(parr_val, r_val),
        "test_spearman": metric(parr_test, r_test),
        "top25_overlap": overlap,
        "epsilon_hat": eps_hat,
        "top25_reduction": reduction,
        "oracle_top25_reduction": oracle_reduction,
        "oracle_gap": oracle_reduction - reduction,
        "selected_hard": selected,
        "neg_pool": ",".join(neg_names),
    }


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
            print(f"seed {seed}: diagnostics {name} ...")
            rows.append(evaluate_dataset(name, *spec, args.root, args.max_windows, args.frac, seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "rank_omega_diagnostics.csv", index=False)

    summary = (
        df.groupby("family", as_index=False)
        .agg(
            runs=("dataset", "count"),
            n_val_mean=("n_val", "mean"),
            omega_mean=("omega", "mean"),
            omega_min=("omega", "min"),
            block_sign_rate=("block_sign_rate", "mean"),
            overlap_mean=("top25_overlap", "mean"),
            epsilon_mean=("epsilon_hat", "mean"),
            test_spearman_mean=("test_spearman", "mean"),
            top25_reduction_mean=("top25_reduction", "mean"),
            oracle_gap_mean=("oracle_gap", "mean"),
        )
        .sort_values("family")
    )
    all_row = pd.DataFrame(
        [
            {
                "family": "All",
                "runs": len(df),
                "n_val_mean": df["n_val"].mean(),
                "omega_mean": df["omega"].mean(),
                "omega_min": df["omega"].min(),
                "block_sign_rate": df["block_sign_rate"].mean(),
                "overlap_mean": df["top25_overlap"].mean(),
                "epsilon_mean": df["epsilon_hat"].mean(),
                "test_spearman_mean": df["test_spearman"].mean(),
                "top25_reduction_mean": df["top25_reduction"].mean(),
                "oracle_gap_mean": df["oracle_gap"].mean(),
            }
        ]
    )
    summary = pd.concat([summary, all_row], ignore_index=True)
    summary.to_csv(outdir / "rank_omega_diagnostics_summary.csv", index=False)

    md = ["# Rank-Error and Omega Diagnostics", ""]
    md.append("Held-out proxy residual audit for the default PARR-rank pool.")
    md.append("")
    md.append(summary.to_markdown(index=False, floatfmt=".4f"))
    (outdir / "rank_omega_diagnostics_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
