import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import risk_scoring_baseline_local as base


class SharedLinearForecaster(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C]. The same temporal projection is shared by all channels.
        return self.proj(x.permute(0, 2, 1)).permute(0, 2, 1)


def standardize_by_train(x, y, train_idx):
    train_values = x[train_idx].reshape(-1, x.shape[-1])
    mean = train_values.mean(axis=0, keepdims=True).astype(np.float32)
    std = (train_values.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    return ((x - mean) / std).astype(np.float32), ((y - mean) / std).astype(np.float32)


def residual_mse(pred, y):
    return ((pred - y) ** 2).mean(axis=(1, 2))


def train_backbone(x, y, train_idx, val_idx, seq_len, pred_len, args, seed):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SharedLinearForecaster(seq_len, pred_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    x_val = torch.from_numpy(x[val_idx]).to(device)
    y_val = torch.from_numpy(y[val_idx]).to(device)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )

    best_state = None
    best_val = float("inf")
    patience_left = args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).detach().cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device, history


def predict(model, device, x, batch_size):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def parr_rank_from_candidates(candidates, parr_names, val_metrics, r_val, r_test, frac):
    parr_val_metrics = {k: val_metrics[k] for k in parr_names}
    selected = min(parr_val_metrics, key=parr_val_metrics.get)
    neg = [k for k, rho in parr_val_metrics.items() if rho < 0] or [selected]
    weights = np.array([-val_metrics[k] for k in neg], dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    ranks_test = np.vstack([base.rank01(candidates[k][1]) for k in neg])
    ranks_val = np.vstack([base.rank01(candidates[k][0]) for k in neg])
    parr_test = weights @ ranks_test
    parr_val = weights @ ranks_val
    return selected, neg, parr_val, parr_test


def parr_plus_from_candidates(candidates, val_metrics):
    neg = [k for k, rho in val_metrics.items() if rho < 0]
    if not neg:
        neg = [min(val_metrics, key=val_metrics.get)]
    weights = np.array([-val_metrics[k] for k in neg], dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    ranks_test = np.vstack([base.rank01(candidates[k][1]) for k in neg])
    ranks_val = np.vstack([base.rank01(candidates[k][0]) for k in neg])
    return neg, weights @ ranks_val, weights @ ranks_test


def evaluate_dataset(name, rel_path, seq_len, pred_len, patch_len, args, seed):
    arr = base.numeric_frame(Path(args.root) / rel_path)
    x_raw, y_raw = base.make_windows(arr, seq_len, pred_len, args.max_windows, seed=seed)
    train_idx, val_idx, test_idx = base.split_temporal(len(x_raw))
    x, y = standardize_by_train(x_raw, y_raw, train_idx)

    model, device, history = train_backbone(x, y, train_idx, val_idx, seq_len, pred_len, args, seed)
    val_pred = predict(model, device, x[val_idx], args.eval_batch_size)
    test_pred = predict(model, device, x[test_idx], args.eval_batch_size)
    r_val = residual_mse(val_pred, y[val_idx])
    r_test = residual_mse(test_pred, y[test_idx])

    features = base.parr_components(x, patch_len)
    aux = base.auxiliary_window_features(x)
    x_val, x_test = features[val_idx], features[test_idx]
    aux_val, aux_test = aux[val_idx], aux[test_idx]
    candidates, parr_names = base.fit_scores(x_val, r_val, x_test, aux_val, aux_test, seed)
    val_metrics = {k: base.metric(v[0], r_val) for k, v in candidates.items()}
    test_metrics = {k: base.metric(v[1], r_test) for k, v in candidates.items()}

    selected, neg, parr_val, parr_test = parr_rank_from_candidates(
        candidates, parr_names, val_metrics, r_val, r_test, args.frac
    )
    candidates["PARR_rank"] = (parr_val, parr_test)
    val_metrics["PARR_rank"] = base.metric(parr_val, r_val)
    test_metrics["PARR_rank"] = base.metric(parr_test, r_test)

    plus_neg, plus_val, plus_test = parr_plus_from_candidates(
        {k: v for k, v in candidates.items() if k != "PARR_rank"},
        {k: v for k, v in val_metrics.items() if k != "PARR_rank"},
    )
    candidates["PARR_plus_strong_rank"] = (plus_val, plus_test)
    val_metrics["PARR_plus_strong_rank"] = base.metric(plus_val, r_val)
    test_metrics["PARR_plus_strong_rank"] = base.metric(plus_test, r_test)

    keep = [
        "PARR_rank",
        "PARR_plus_strong_rank",
        "ridge_risk",
        "full_ridge_risk",
        "gbdt_risk",
        "mlp_risk",
        "knn_mean_risk",
        "knn_q90_risk",
        "mahalanobis_ood",
        "full_mahalanobis_ood",
        "pca_reconstruction_ood",
        "random",
    ]
    rng = np.random.default_rng(seed)
    candidates["random"] = (rng.normal(size=len(r_val)), rng.normal(size=len(r_test)))
    val_metrics["random"] = np.nan
    test_metrics["random"] = base.metric(candidates["random"][1], r_test)

    rows = []
    for scorer in keep:
        if scorer not in candidates:
            continue
        score = candidates[scorer][1]
        rows.append(
            {
                "dataset": name,
                "seed": seed,
                "backbone": "SharedLinear",
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "backbone_val_mse": float(r_val.mean()),
                "backbone_test_mse": float(r_test.mean()),
                "best_epoch": int(min(history, key=lambda h: h["val_loss"])["epoch"]),
                "scorer": scorer,
                "val_spearman": val_metrics.get(scorer, np.nan),
                "test_spearman": test_metrics.get(scorer, np.nan),
                "top25_reduction": base.top_reduction(score, r_test, args.frac),
                "risk_coverage_auc": base.coverage_auc(score, r_test),
                "selected_hard": scorer == selected,
                "parr_neg_pool": ",".join(neg),
                "strong_neg_count": len(plus_neg),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="local_experiments/results_frozen_shared_linear")
    parser.add_argument("--max-windows", type=int, default=5000)
    parser.add_argument("--frac", type=float, default=0.25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    all_rows = []
    for seed in args.seeds:
        for name, spec in base.DATASETS.items():
            path = Path(args.root) / spec[0]
            if not path.exists():
                print(f"skip {name}: missing {path}")
                continue
            print(f"seed {seed}: frozen SharedLinear {name} ...", flush=True)
            all_rows.extend(evaluate_dataset(name, *spec, args, seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(outdir / "frozen_shared_linear_strong_baselines.csv", index=False)

    per_dataset = (
        df.groupby(["scorer", "dataset"], as_index=False)
        .agg(
            top25_mean=("top25_reduction", "mean"),
            auc_mean=("risk_coverage_auc", "mean"),
            positive_rate=("top25_reduction", lambda x: (x > 0).mean()),
        )
    )
    per_dataset.to_csv(outdir / "frozen_shared_linear_per_dataset.csv", index=False)
    summary = (
        per_dataset.groupby("scorer")
        .agg(
            mean_top25=("top25_mean", "mean"),
            min_top25=("top25_mean", "min"),
            mean_auc=("auc_mean", "mean"),
            min_auc=("auc_mean", "min"),
            positive_cases=("top25_mean", lambda x: (x > 0).sum()),
            num_cases=("top25_mean", "count"),
        )
        .sort_values(["mean_top25", "mean_auc"], ascending=False)
    )
    summary.to_csv(outdir / "frozen_shared_linear_summary.csv")

    md = ["# Frozen SharedLinear Strong Risk Baselines", ""]
    md.append("All risk scorers are calibrated on the same frozen SharedLinear validation residual target.")
    md.append("")
    md.append(summary.to_markdown(floatfmt=".4f"))
    md.append("")
    md.append("## Per-Dataset Top-25 Reduction")
    md.append("")
    pivot = per_dataset.pivot(index="scorer", columns="dataset", values="top25_mean")
    md.append(pivot.to_markdown(floatfmt=".4f"))
    (outdir / "frozen_shared_linear_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    main()
