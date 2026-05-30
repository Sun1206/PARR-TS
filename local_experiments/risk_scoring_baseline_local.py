import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import rankdata, spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


DATASETS = {
    "ETTh1": ("datasets/ETT-small/ETTh1.csv", 96, 96, 16),
    "ETTh2": ("datasets/ETT-small/ETTh2.csv", 96, 96, 16),
    "ETTm1": ("datasets/ETT-small/ETTm1.csv", 96, 96, 16),
    "ETTm2": ("data_cache/ETTm2.csv", 96, 96, 16),
    "Weather": ("data_cache/weather.csv", 96, 96, 16),
    "Exchange": ("data_cache/exchange_rate.csv", 96, 96, 16),
    "Electricity": ("data_cache/electricity.csv", 96, 96, 16),
    "Illness": ("data_cache/national_illness.csv", 36, 24, 12),
}


def numeric_frame(path):
    df = pd.read_csv(path)
    cols = []
    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().mean() > 0.95:
            cols.append(col)
    arr = df[cols].apply(pd.to_numeric, errors="coerce").ffill().bfill().to_numpy(dtype=np.float32)
    std = arr.std(axis=0)
    keep = std > 1e-8
    return arr[:, keep]


def make_windows(arr, seq_len, pred_len, max_windows, seed=0):
    n = len(arr) - seq_len - pred_len + 1
    if n <= 0:
        raise ValueError("not enough rows")
    idx = np.arange(n)
    if max_windows and n > max_windows:
        rng = np.random.default_rng(seed)
        # Preserve temporal coverage while keeping runtime bounded.
        idx = np.sort(rng.choice(idx, size=max_windows, replace=False))
    x = np.stack([arr[i : i + seq_len] for i in idx])
    y = np.stack([arr[i + seq_len : i + seq_len + pred_len] for i in idx])
    return x, y


def naive_residual(x, y):
    pred = np.repeat(x[:, -1:, :], y.shape[1], axis=1)
    return ((pred - y) ** 2).mean(axis=(1, 2))


def parr_components(x, patch_len):
    n, seq_len, channels = x.shape
    m = seq_len // patch_len
    x = x[:, : m * patch_len]
    patches = x.reshape(n, m, patch_len, channels)

    freq = np.fft.rfft(patches, axis=2)
    energy = (np.abs(freq) ** 2).mean(axis=-1)
    prob = energy / (energy.sum(axis=-1, keepdims=True) + 1e-8)
    spectral = -(prob * np.log(prob + 1e-8)).sum(axis=-1) / (np.log(prob.shape[-1]) + 1e-8)

    if energy.shape[-1] <= 2:
        period = np.zeros((n, m), dtype=np.float32)
    else:
        dominant = energy[:, :, 1:].argmax(axis=-1).astype(np.float32) + 1.0
        ref = dominant.mean(axis=1, keepdims=True)
        period = np.clip(np.abs(dominant - ref) / (np.abs(ref) + 1e-8), 0.0, 1.0)

    pad = np.pad(patches, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="edge")
    smooth = (pad[:, :, :-2] + pad[:, :, 1:-1] + pad[:, :, 2:]) / 3.0
    smooth_res = ((patches - smooth) ** 2).mean(axis=(2, 3)) / ((patches**2).mean(axis=(2, 3)) + 1e-8)
    smooth_res = np.clip(smooth_res, 0.0, 1.0)

    profile = patches.mean(axis=2)
    ref_profile = profile.mean(axis=1, keepdims=True)
    dot = (profile * ref_profile).sum(axis=-1)
    norm = np.linalg.norm(profile, axis=-1) * np.linalg.norm(ref_profile, axis=-1) + 1e-8
    channel = np.clip(1.0 - dot / norm, 0.0, 1.0)

    return np.stack(
        [
            spectral.mean(axis=1),
            period.mean(axis=1),
            smooth_res.mean(axis=1),
            channel.mean(axis=1),
        ],
        axis=1,
    )


def auxiliary_window_features(x):
    diff = x[:, 1:] - x[:, :-1]
    q = max(2, x.shape[1] // 4)
    recent_diff = diff[:, -q:]
    first = x[:, : x.shape[1] // 2]
    second = x[:, x.shape[1] // 2 :]
    scale = (x**2).mean(axis=(1, 2)) + 1e-8

    whole_vol = (diff**2).mean(axis=(1, 2))
    recent_vol = (recent_diff**2).mean(axis=(1, 2))
    last_jump = ((x[:, -1] - x[:, -2]) ** 2).mean(axis=1)
    level_shift = ((second.mean(axis=1) - first.mean(axis=1)) ** 2).mean(axis=1) / scale
    amplitude = x.std(axis=1).mean(axis=1)
    channel_dispersion = x.mean(axis=1).std(axis=1)
    trend = np.abs(x[:, -q:].mean(axis=1) - x[:, :q].mean(axis=1)).mean(axis=1) / (np.sqrt(scale) + 1e-8)

    return np.stack(
        [
            whole_vol,
            recent_vol,
            last_jump,
            level_shift,
            amplitude,
            channel_dispersion,
            trend,
        ],
        axis=1,
    )


def split_temporal(n):
    a = int(n * 0.7)
    b = int(n * 0.8)
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)


def metric(score, residual):
    val = spearmanr(score, residual).statistic
    return float(val) if not np.isnan(val) else 0.0


def rank01(score):
    if len(score) <= 1:
        return np.zeros_like(score, dtype=np.float64)
    return (rankdata(score, method="average") - 1.0) / (len(score) - 1.0)


def top_reduction(score, residual, frac=0.25):
    order = np.argsort(score)
    k = max(1, int(frac * len(order)))
    selected = residual[order[-k:]].mean()
    overall = residual.mean()
    return float(1.0 - selected / overall)


def coverage_auc(score, residual, coverages=(0.1, 0.25, 0.5, 0.75)):
    return float(np.mean([top_reduction(score, residual, c) for c in coverages]))


def fit_scores(x_val, r_val, x_test, aux_val, aux_test, seed):
    parr_scaler = StandardScaler().fit(x_val)
    zv = parr_scaler.transform(x_val)
    zt = parr_scaler.transform(x_test)
    full_val = np.concatenate([x_val, aux_val], axis=1)
    full_test = np.concatenate([x_test, aux_test], axis=1)
    full_scaler = StandardScaler().fit(full_val)
    fv = full_scaler.transform(full_val)
    ft = full_scaler.transform(full_test)
    candidates = {}
    parr_candidate_names = set()

    fixed_beta = np.array([-1.0, 1.0, -1.0, -1.0])
    candidates["fixed_formula"] = (-(x_val @ fixed_beta), -(x_test @ fixed_beta))
    parr_candidate_names.add("fixed_formula")

    directions = []
    for i, name in enumerate(["spectral", "period", "smooth", "channel"]):
        corr = metric(zv[:, i], r_val)
        direction = 1.0 if corr <= 0 else -1.0
        directions.append(direction)
        candidates[f"single_{name}"] = (direction * zv[:, i], direction * zt[:, i])
        parr_candidate_names.add(f"single_{name}")

    directions = np.array(directions)
    candidates["sign_linear"] = (zv @ directions, zt @ directions)
    parr_candidate_names.add("sign_linear")

    ridge = Ridge(alpha=1.0).fit(zv, r_val)
    candidates["ridge_risk"] = (-ridge.predict(zv), -ridge.predict(zt))
    parr_candidate_names.add("ridge_risk")

    full_ridge = Ridge(alpha=1.0).fit(fv, r_val)
    candidates["full_ridge_risk"] = (-full_ridge.predict(fv), -full_ridge.predict(ft))

    gb = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.04,
        max_leaf_nodes=8,
        l2_regularization=0.05,
        random_state=seed,
    ).fit(fv, r_val)
    candidates["gbdt_risk"] = (-gb.predict(fv), -gb.predict(ft))

    mlp = MLPRegressor(
        hidden_layer_sizes=(24,),
        activation="relu",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=800,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=seed,
    ).fit(fv, r_val)
    candidates["mlp_risk"] = (-mlp.predict(fv), -mlp.predict(ft))

    k = min(50, len(fv))
    nn = NearestNeighbors(n_neighbors=k).fit(fv)
    _, val_nn = nn.kneighbors(fv)
    _, test_nn = nn.kneighbors(ft)
    candidates["knn_mean_risk"] = (-r_val[val_nn].mean(axis=1), -r_val[test_nn].mean(axis=1))
    candidates["knn_q90_risk"] = (
        -np.quantile(r_val[val_nn], 0.9, axis=1),
        -np.quantile(r_val[test_nn], 0.9, axis=1),
    )

    lw = LedoitWolf().fit(zv)
    dist_val = lw.mahalanobis(zv)
    dist_test = lw.mahalanobis(zt)
    candidates["mahalanobis_ood"] = (-dist_val, -dist_test)

    full_lw = LedoitWolf().fit(fv)
    candidates["full_mahalanobis_ood"] = (-full_lw.mahalanobis(fv), -full_lw.mahalanobis(ft))

    n_comp = min(4, fv.shape[1], max(1, len(fv) - 1))
    pca = PCA(n_components=n_comp, random_state=seed).fit(fv)
    val_rec = pca.inverse_transform(pca.transform(fv))
    test_rec = pca.inverse_transform(pca.transform(ft))
    candidates["pca_reconstruction_ood"] = (
        -((fv - val_rec) ** 2).mean(axis=1),
        -((ft - test_rec) ** 2).mean(axis=1),
    )

    candidates["variance"] = (-full_val.var(axis=1), -full_test.var(axis=1))
    aux_names = [
        "rolling_volatility",
        "recent_volatility",
        "last_jump",
        "level_shift",
        "amplitude",
        "channel_dispersion",
        "trend_shift",
    ]
    for offset, name in enumerate(aux_names, start=x_val.shape[1]):
        corr = metric(fv[:, offset], r_val)
        direction = 1.0 if corr <= 0 else -1.0
        candidates[name] = (direction * fv[:, offset], direction * ft[:, offset])

    return candidates, parr_candidate_names


def evaluate_dataset(name, rel_path, seq_len, pred_len, patch_len, args):
    arr = numeric_frame(Path(args.root) / rel_path)
    x, y = make_windows(arr, seq_len, pred_len, args.max_windows, args.seed)
    residual = naive_residual(x, y)
    features = parr_components(x, patch_len)
    aux_features = auxiliary_window_features(x)
    _, val_idx, test_idx = split_temporal(len(residual))

    x_val, r_val = features[val_idx], residual[val_idx]
    x_test, r_test = features[test_idx], residual[test_idx]
    aux_val, aux_test = aux_features[val_idx], aux_features[test_idx]
    candidates, parr_candidate_names = fit_scores(x_val, r_val, x_test, aux_val, aux_test, args.seed)
    val_metrics = {k: metric(v[0], r_val) for k, v in candidates.items()}
    test_metrics = {k: metric(v[1], r_test) for k, v in candidates.items()}
    reductions = {k: top_reduction(v[1], r_test, args.frac) for k, v in candidates.items()}
    aucs = {k: coverage_auc(v[1], r_test) for k, v in candidates.items()}

    parr_val_metrics = {k: v for k, v in val_metrics.items() if k in parr_candidate_names}
    selected = min(parr_val_metrics, key=parr_val_metrics.get)
    neg = [k for k, rho in parr_val_metrics.items() if rho < 0] or [selected]
    weights = np.array([-val_metrics[k] for k in neg], dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    ranks_test = np.vstack([rank01(candidates[k][1]) for k in neg])
    ranks_val = np.vstack([rank01(candidates[k][0]) for k in neg])
    parr_test = weights @ ranks_test
    parr_val = weights @ ranks_val
    reductions["PARR_full_rank"] = top_reduction(parr_test, r_test, args.frac)
    aucs["PARR_full_rank"] = coverage_auc(parr_test, r_test)
    test_metrics["PARR_full_rank"] = metric(parr_test, r_test)
    val_metrics["PARR_full_rank"] = metric(parr_val, r_val)

    all_neg = [k for k, rho in val_metrics.items() if rho < 0 and k != "PARR_full_rank"]
    if all_neg:
        all_weights = np.array([-val_metrics[k] for k in all_neg], dtype=np.float64)
        all_weights = all_weights / (all_weights.sum() + 1e-12)
        all_rank_test = np.vstack([rank01(candidates[k][1]) for k in all_neg])
        all_rank_val = np.vstack([rank01(candidates[k][0]) for k in all_neg])
        parr_plus_test = all_weights @ all_rank_test
        parr_plus_val = all_weights @ all_rank_val
        reductions["PARR_plus_strong_rank"] = top_reduction(parr_plus_test, r_test, args.frac)
        aucs["PARR_plus_strong_rank"] = coverage_auc(parr_plus_test, r_test)
        test_metrics["PARR_plus_strong_rank"] = metric(parr_plus_test, r_test)
        val_metrics["PARR_plus_strong_rank"] = metric(parr_plus_val, r_val)

    rng = np.random.default_rng(args.seed)
    random_scores = rng.normal(size=len(r_test))
    reductions["random"] = top_reduction(random_scores, r_test, args.frac)
    aucs["random"] = coverage_auc(random_scores, r_test)
    test_metrics["random"] = metric(random_scores, r_test)
    val_metrics["random"] = np.nan

    rows = []
    for scorer in sorted(reductions):
        rows.append(
            {
                "dataset": name,
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "scorer": scorer,
                "val_spearman": val_metrics.get(scorer, np.nan),
                "test_spearman": test_metrics.get(scorer, np.nan),
                "top25_reduction": reductions[scorer],
                "risk_coverage_auc": aucs[scorer],
                "selected_hard": scorer == selected,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="local_experiments/results")
    parser.add_argument("--max-windows", type=int, default=12000)
    parser.add_argument("--frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    all_rows = []
    for name, spec in DATASETS.items():
        path = Path(args.root) / spec[0]
        if not path.exists():
            print(f"skip {name}: missing {path}")
            continue
        print(f"running {name} ...")
        all_rows.extend(evaluate_dataset(name, *spec, args))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(outdir / "local_risk_scoring_baselines.csv", index=False)

    pivot = df.pivot_table(index="scorer", columns="dataset", values="top25_reduction", aggfunc="mean")
    summary = pd.DataFrame(
        {
            "mean_top25": pivot.mean(axis=1),
            "min_top25": pivot.min(axis=1),
            "positive_cases": (pivot > 0).sum(axis=1),
            "num_cases": pivot.notna().sum(axis=1),
        }
    ).sort_values(["mean_top25", "min_top25"], ascending=False)
    summary.to_csv(outdir / "local_risk_scoring_summary.csv")

    md = ["# Local Risk-Scoring Baseline Experiment", ""]
    md.append("Proxy target: naive persistence forecasting residual on public CSV windows.")
    md.append("")
    md.append(summary.to_markdown(floatfmt=".4f"))
    md.append("")
    md.append("## Per-Dataset Top-25 Reduction")
    md.append("")
    md.append(pivot.to_markdown(floatfmt=".4f"))
    (outdir / "local_risk_scoring_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
