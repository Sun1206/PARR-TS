import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import risk_scoring_baseline_local as base


PARR_NAMES = ["spectral", "period", "smooth", "channel"]
AUX_NAMES = [
    "rolling_volatility",
    "recent_volatility",
    "last_jump",
    "level_shift",
    "amplitude",
    "channel_dispersion",
    "trend_shift",
]


def metric(score, residual):
    if len(score) < 2 or np.std(score) <= 1e-12 or np.std(residual) <= 1e-12:
        return 0.0
    value = spearmanr(score, residual).statistic
    return float(value) if not np.isnan(value) else 0.0


def rank_redundancy(score, core_scores):
    vals = [abs(metric(score, core_scores[:, i])) for i in range(core_scores.shape[1])]
    return float(max(vals)) if vals else np.nan


def block_sign_stability(score, residual, blocks=4):
    full_corr = metric(score, residual)
    direction = 1.0 if full_corr <= 0 else -1.0
    oriented = direction * score
    values = []
    for idx in np.array_split(np.arange(len(residual)), blocks):
        if len(idx) < 4:
            continue
        values.append(metric(oriented[idx], residual[idx]) < 0)
    return float(np.mean(values)) if values else np.nan


def ood_scores(parr_val, parr_test, full_val, full_test, seed):
    scores = {}
    lw = LedoitWolf().fit(parr_val)
    scores["parr_mahalanobis"] = (-lw.mahalanobis(parr_val), -lw.mahalanobis(parr_test))
    full_lw = LedoitWolf().fit(full_val)
    scores["full_mahalanobis"] = (-full_lw.mahalanobis(full_val), -full_lw.mahalanobis(full_test))
    n_comp = min(4, full_val.shape[1], max(1, len(full_val) - 1))
    pca = PCA(n_components=n_comp, random_state=seed).fit(full_val)
    val_rec = pca.inverse_transform(pca.transform(full_val))
    test_rec = pca.inverse_transform(pca.transform(full_test))
    scores["pca_reconstruction"] = (
        -((full_val - val_rec) ** 2).mean(axis=1),
        -((full_test - test_rec) ** 2).mean(axis=1),
    )
    return scores


def evaluate_dataset(name, rel_path, seq_len, pred_len, patch_len, root, max_windows, seed):
    arr = base.numeric_frame(Path(root) / rel_path)
    x, y = base.make_windows(arr, seq_len, pred_len, max_windows, seed=seed)
    residual = base.naive_residual(x, y)
    _, val_idx, test_idx = base.split_temporal(len(residual))

    start = time.perf_counter()
    parr = base.parr_components(x, patch_len)
    parr_sec = time.perf_counter() - start
    start = time.perf_counter()
    aux = base.auxiliary_window_features(x)
    aux_sec = time.perf_counter() - start

    parr_val_raw, parr_test_raw = parr[val_idx], parr[test_idx]
    aux_val_raw, aux_test_raw = aux[val_idx], aux[test_idx]
    r_val = residual[val_idx]

    parr_scaler = StandardScaler().fit(parr_val_raw)
    parr_val = parr_scaler.transform(parr_val_raw)
    parr_test = parr_scaler.transform(parr_test_raw)
    full_val_raw = np.concatenate([parr_val_raw, aux_val_raw], axis=1)
    full_test_raw = np.concatenate([parr_test_raw, aux_test_raw], axis=1)
    full_scaler = StandardScaler().fit(full_val_raw)
    full_val = full_scaler.transform(full_val_raw)
    full_test = full_scaler.transform(full_test_raw)
    aux_val = full_val[:, len(PARR_NAMES) :]
    aux_test = full_test[:, len(PARR_NAMES) :]

    start = time.perf_counter()
    ood = ood_scores(parr_val, parr_test, full_val, full_test, seed)
    ood_sec = time.perf_counter() - start

    rows = []
    core = parr_val
    for i, desc in enumerate(PARR_NAMES):
        score = parr_val[:, i]
        others = np.delete(core, i, axis=1)
        rows.append(
            row(name, seed, "PARR core", desc, score, r_val, rank_redundancy(score, others), parr_sec / len(PARR_NAMES))
        )
    for i, desc in enumerate(AUX_NAMES):
        score = aux_val[:, i]
        rows.append(
            row(name, seed, "Auxiliary local", desc, score, r_val, rank_redundancy(score, core), aux_sec / len(AUX_NAMES))
        )
    for desc, (score_val, _) in ood.items():
        rows.append(
            row(name, seed, "OOD/distance", desc, score_val, r_val, rank_redundancy(score_val, core), ood_sec / len(ood))
        )
    return rows


def row(dataset, seed, group, descriptor, score, residual, redundancy, cpu_sec):
    corr = metric(score, residual)
    return {
        "dataset": dataset,
        "family": "ETT" if dataset.startswith("ETT") else dataset,
        "seed": seed,
        "group": group,
        "descriptor": descriptor,
        "val_spearman": corr,
        "relevance_abs": abs(corr),
        "negative_mass": max(-corr, 0.0),
        "block_sign_stability": block_sign_stability(score, residual),
        "redundancy_to_core": redundancy,
        "cpu_sec_per_descriptor": cpu_sec,
    }


def summarize(df):
    summary = (
        df.groupby("group", as_index=False)
        .agg(
            descriptors=("descriptor", "nunique"),
            relevance=("relevance_abs", "mean"),
            neg_mass=("negative_mass", "mean"),
            block_sign=("block_sign_stability", "mean"),
            redundancy=("redundancy_to_core", "mean"),
            cpu_sec=("cpu_sec_per_descriptor", "mean"),
        )
        .sort_values("group")
    )
    return summary


def summarize_descriptors(df):
    return (
        df.groupby(["group", "descriptor"], as_index=False)
        .agg(
            relevance=("relevance_abs", "mean"),
            neg_mass=("negative_mass", "mean"),
            block_sign=("block_sign_stability", "mean"),
            redundancy=("redundancy_to_core", "mean"),
            cpu_sec=("cpu_sec_per_descriptor", "mean"),
        )
        .sort_values(["group", "relevance"], ascending=[True, False])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="local_experiments/results_multiseed")
    parser.add_argument("--max-windows", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 23])
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        for name, spec in base.DATASETS.items():
            path = Path(args.root) / spec[0]
            if not path.exists():
                print(f"skip {name}: missing {path}")
                continue
            print(f"seed {seed}: descriptor audit {name} ...")
            rows.extend(evaluate_dataset(name, *spec, args.root, args.max_windows, seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "descriptor_selection_audit.csv", index=False)
    summary = summarize(df)
    descriptor_summary = summarize_descriptors(df)
    summary.to_csv(outdir / "descriptor_selection_audit_summary.csv", index=False)
    descriptor_summary.to_csv(outdir / "descriptor_selection_audit_by_descriptor.csv", index=False)

    md = ["# Descriptor Selection Audit", ""]
    md.append("Group-level audit over three local proxy seeds.")
    md.append("")
    md.append(summary.to_markdown(index=False, floatfmt=".4f"))
    md.append("")
    md.append("## Descriptor-level")
    md.append("")
    md.append(descriptor_summary.to_markdown(index=False, floatfmt=".4f"))
    (outdir / "descriptor_selection_audit_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
