import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import risk_scoring_baseline_local as base
from rank_omega_diagnostics import fit_parr_pool, metric


def integrated_neff(x, max_lag=80):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return float(len(x)), 0.0
    rhos = []
    for lag in range(1, min(max_lag, len(x) - 1) + 1):
        rho = float(np.dot(x[:-lag], x[lag:]) / denom)
        if rho <= 0:
            break
        rhos.append(rho)
    tau = 1.0 + 2.0 * sum(rhos)
    return float(len(x) / max(tau, 1.0)), float(tau)


def parr_rank_score(x_val, r_val, x_test):
    candidates = fit_parr_pool(x_val, r_val, x_test)
    val_corr = {name: metric(scores[0], r_val) for name, scores in candidates.items()}
    selected = min(val_corr, key=val_corr.get)
    neg = [name for name, rho in val_corr.items() if rho < 0] or [selected]
    weights = np.array([max(-val_corr[name], 0.0) for name in neg], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(neg), dtype=np.float64)
    weights /= weights.sum()
    ranks_val = np.vstack([base.rank01(candidates[name][0]) for name in neg])
    ranks_test = np.vstack([base.rank01(candidates[name][1]) for name in neg])
    return weights @ ranks_val, weights @ ranks_test, selected, ",".join(neg)


def family_of(name):
    return "ETT" if name.startswith("ETT") else name


def evaluate_dataset(name, spec, root, max_windows, seed):
    arr = base.numeric_frame(Path(root) / spec[0])
    x, y = base.make_windows(arr, spec[1], spec[2], max_windows, seed)
    residual = base.naive_residual(x, y)
    features = base.parr_components(x, spec[3])
    _, val_idx, test_idx = base.split_temporal(len(residual))
    parr_val, parr_test, selected, neg = parr_rank_score(
        features[val_idx], residual[val_idx], features[test_idx]
    )
    residual_neff, residual_tau = integrated_neff(base.rank01(residual[val_idx]))
    score_neff, score_tau = integrated_neff(base.rank01(parr_val))
    return {
        "dataset": name,
        "family": family_of(name),
        "channels": arr.shape[1],
        "windows": len(residual),
        "n_val": len(val_idx),
        "residual_neff": residual_neff,
        "residual_neff_ratio": residual_neff / len(val_idx),
        "residual_tau": residual_tau,
        "score_neff": score_neff,
        "score_neff_ratio": score_neff / len(val_idx),
        "score_tau": score_tau,
        "selected": selected,
        "neg_pool": neg,
        "top25_reduction": base.top_reduction(parr_test, residual[test_idx]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="local_experiments/results_multiseed")
    parser.add_argument("--max-windows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = []
    for name, spec in base.DATASETS.items():
        path = Path(args.root) / spec[0]
        if not path.exists():
            print(f"skip {name}: missing {path}")
            continue
        print(f"effective sample diagnostic {name} ...")
        rows.append(evaluate_dataset(name, spec, args.root, args.max_windows, args.seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "effective_sample_diagnostics.csv", index=False)
    family = (
        df.groupby("family", as_index=False)
        .agg(
            runs=("dataset", "count"),
            n_val=("n_val", "mean"),
            residual_neff_ratio=("residual_neff_ratio", "mean"),
            score_neff_ratio=("score_neff_ratio", "mean"),
            top25_reduction=("top25_reduction", "mean"),
        )
        .sort_values("family")
    )
    family.to_csv(outdir / "effective_sample_diagnostics_summary.csv", index=False)
    print(df.to_string(index=False))
    print(family.to_string(index=False))


if __name__ == "__main__":
    main()
