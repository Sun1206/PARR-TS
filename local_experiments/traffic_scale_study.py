import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import risk_scoring_baseline_local as base


def sampled_indices(n_rows, seq_len, pred_len, max_windows, seed):
    n = n_rows - seq_len - pred_len + 1
    if n <= 0:
        raise ValueError("not enough rows for Traffic windows")
    idx = np.arange(n)
    if max_windows and n > max_windows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_windows, replace=False))
    return idx


def residual_for_batch(x, y):
    pred = np.repeat(x[:, -1:, :], y.shape[1], axis=1)
    return ((pred - y) ** 2).mean(axis=(1, 2))


def materialize_features(arr, idx, seq_len, pred_len, patch_len, batch_size):
    residuals, parr, aux = [], [], []
    t_window = 0.0
    t_feature = 0.0
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start : start + batch_size]
        t0 = time.perf_counter()
        x = np.stack([arr[i : i + seq_len] for i in batch_idx])
        y = np.stack([arr[i + seq_len : i + seq_len + pred_len] for i in batch_idx])
        residuals.append(residual_for_batch(x, y))
        t1 = time.perf_counter()
        parr.append(base.parr_components(x, patch_len))
        aux.append(base.auxiliary_window_features(x))
        t2 = time.perf_counter()
        t_window += t1 - t0
        t_feature += t2 - t1
    return (
        np.concatenate(residuals),
        np.vstack(parr),
        np.vstack(aux),
        t_window,
        t_feature,
    )


def metric(score, residual):
    if len(score) < 2 or np.std(score) <= 1e-12 or np.std(residual) <= 1e-12:
        return 0.0
    value = spearmanr(score, residual).statistic
    return float(value) if not np.isnan(value) else 0.0


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


def add_proxy_scores(candidates, full_val, full_test, r_val, seed):
    scaler = StandardScaler().fit(full_val)
    fv = scaler.transform(full_val)
    ft = scaler.transform(full_test)

    # Energy-style input score: high standardized feature energy is treated as risky.
    candidates["energy_style_ood"] = (-(fv**2).mean(axis=1), -(ft**2).mean(axis=1))

    # Shapelet/dictionary-style proxy in descriptor space: proximity to high-risk
    # calibration descriptors indicates risk, without inspecting test residuals.
    q = np.quantile(r_val, 0.75)
    high = fv[r_val >= q]
    if len(high) < 2:
        high = fv
    nn = NearestNeighbors(n_neighbors=min(10, len(high))).fit(high)
    val_dist, _ = nn.kneighbors(fv)
    test_dist, _ = nn.kneighbors(ft)
    candidates["shapelet_style_highrisk_distance"] = (
        val_dist.mean(axis=1),
        test_dist.mean(axis=1),
    )

    # Training-dynamics-style proxy: disagreement of simple one-step extrapolation
    # surrogates, available when training checkpoints are not stored.
    rng = np.random.default_rng(seed)
    jitter = rng.normal(scale=1e-6, size=len(fv))
    candidates["dynamics_disagreement_proxy"] = (
        -(np.var(fv[:, :4], axis=1) + jitter),
        -np.var(ft[:, :4], axis=1),
    )


def evaluate_scores(features, aux, residual, seed, frac):
    _, val_idx, test_idx = base.split_temporal(len(residual))
    x_val, x_test = features[val_idx], features[test_idx]
    a_val, a_test = aux[val_idx], aux[test_idx]
    r_val, r_test = residual[val_idx], residual[test_idx]
    candidates, parr_names = base.fit_scores(x_val, r_val, x_test, a_val, a_test, seed)
    full_val = np.concatenate([x_val, a_val], axis=1)
    full_test = np.concatenate([x_test, a_test], axis=1)
    add_proxy_scores(candidates, full_val, full_test, r_val, seed)

    val_metrics = {k: metric(v[0], r_val) for k, v in candidates.items()}
    rows = []
    for name, (s_val, s_test) in candidates.items():
        rows.append(
            {
                "method": name,
                "val_spearman": val_metrics[name],
                "test_spearman": metric(s_test, r_test),
                "top25": base.top_reduction(s_test, r_test, frac),
                "auc": base.coverage_auc(s_test, r_test),
            }
        )

    selected = min(parr_names, key=lambda k: val_metrics[k])
    neg = [k for k in parr_names if val_metrics[k] < 0] or [selected]
    weights = np.array([max(-val_metrics[k], 0.0) for k in neg])
    if weights.sum() <= 0:
        weights = np.ones(len(neg))
    weights = weights / weights.sum()
    parr_val = weights @ np.vstack([base.rank01(candidates[k][0]) for k in neg])
    parr_test = weights @ np.vstack([base.rank01(candidates[k][1]) for k in neg])
    rows.append(
        {
            "method": "PARR_full_rank",
            "val_spearman": metric(parr_val, r_val),
            "test_spearman": metric(parr_test, r_test),
            "top25": base.top_reduction(parr_test, r_test, frac),
            "auc": base.coverage_auc(parr_test, r_test),
        }
    )

    selected_val, selected_test = candidates[selected]
    delta = abs(selected_test.mean() - selected_val.mean()) / (selected_val.std() + 1e-8)
    neff_r, tau_r = integrated_neff(base.rank01(r_val))
    neff_s, tau_s = integrated_neff(base.rank01(parr_val))
    gate = len(r_val) >= 200 and delta <= 0.60
    diagnostics = {
        "n_val": len(r_val),
        "n_test": len(r_test),
        "selected_hard": selected,
        "negative_pool": ",".join(neg),
        "delta_score": float(delta),
        "gate_gamma_0p60": "apply" if gate else "abstain",
        "residual_neff": neff_r,
        "residual_tau": tau_r,
        "score_neff": neff_s,
        "score_tau": tau_s,
    }
    return pd.DataFrame(rows).sort_values("top25", ascending=False), diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="local_experiments/results_traffic")
    parser.add_argument("--path", default="data_cache/traffic.csv")
    parser.add_argument("--max-windows", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frac", type=float, default=0.25)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = Path(args.root) / args.path
    t0 = time.perf_counter()
    arr = base.numeric_frame(path)
    t1 = time.perf_counter()
    idx = sampled_indices(len(arr), 96, 96, args.max_windows, args.seed)
    residual, features, aux, t_window, t_feature = materialize_features(
        arr, idx, 96, 96, 16, args.batch_size
    )
    t2 = time.perf_counter()
    table, diagnostics = evaluate_scores(features, aux, residual, args.seed, args.frac)
    t3 = time.perf_counter()

    table.insert(0, "dataset", "Traffic")
    table.to_csv(outdir / "traffic_scale_scores.csv", index=False)
    diag = {
        "dataset": "Traffic",
        "channels": arr.shape[1],
        "rows": len(arr),
        "windows": len(idx),
        "load_sec": t1 - t0,
        "window_residual_sec": t_window,
        "parr_feature_sec": t_feature,
        "scorer_fit_eval_sec": t3 - t2,
        "total_sec": t3 - t0,
        "parr_feature_ms_per_window": 1000.0 * t_feature / len(idx),
        **diagnostics,
    }
    pd.DataFrame([diag]).to_csv(outdir / "traffic_scale_diagnostics.csv", index=False)
    print(pd.DataFrame([diag]).to_string(index=False))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
