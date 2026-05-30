import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import frozen_shared_linear_strong_baselines as shared
import risk_scoring_baseline_local as base


class PatchTSTForecaster(nn.Module):
    def __init__(self, root, seq_len, pred_len, enc_in, args):
        super().__init__()
        import sys

        tslib = Path(root) / "Time-Series-Library"
        sys.path.insert(0, str(tslib.resolve()))
        from models.PatchTST import Model

        cfg = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=args.d_model,
            d_ff=args.d_ff,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            factor=args.factor,
            dropout=args.dropout,
            activation="gelu",
        )
        self.model = Model(cfg, patch_len=args.patch_len, stride=args.stride)

    def forward(self, x):
        return self.model(x, None, None, None)


def train_patchtst(x, y, train_idx, val_idx, args, seed):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PatchTSTForecaster(args.root, x.shape[1], y.shape[1], x.shape[2], args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    x_val = torch.from_numpy(x[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    best_state = None
    best_val = float("inf")
    patience_left = args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(x_val), args.eval_batch_size):
                xb = x_val[start : start + args.eval_batch_size].to(device)
                yb = y_val[start : start + args.eval_batch_size].to(device)
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    val_losses.append(float(loss_fn(model(xb), yb).detach().cpu()))
        val_loss = float(np.mean(val_losses))
        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:02d} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
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


def predict(model, device, x, batch_size, amp):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                preds.append(model(xb).detach().cpu().float().numpy())
    return np.concatenate(preds, axis=0)


def evaluate(args):
    rel_path, seq_len, pred_len, patch_len = base.DATASETS[args.dataset]
    arr = base.numeric_frame(Path(args.root) / rel_path)
    max_windows = None if args.max_windows <= 0 else args.max_windows
    x_raw, y_raw = base.make_windows(arr, seq_len, pred_len, max_windows, seed=args.seed)
    train_idx, val_idx, test_idx = base.split_temporal(len(x_raw))
    x, y = shared.standardize_by_train(x_raw, y_raw, train_idx)
    args.patch_len = patch_len

    model, device, history = train_patchtst(x, y, train_idx, val_idx, args, args.seed)
    val_pred = predict(model, device, x[val_idx], args.eval_batch_size, args.amp)
    test_pred = predict(model, device, x[test_idx], args.eval_batch_size, args.amp)
    r_val = shared.residual_mse(val_pred, y[val_idx])
    r_test = shared.residual_mse(test_pred, y[test_idx])

    features = base.parr_components(x, args.patch_len)
    aux = base.auxiliary_window_features(x)
    x_val, x_test = features[val_idx], features[test_idx]
    aux_val, aux_test = aux[val_idx], aux[test_idx]
    candidates, parr_names = base.fit_scores(x_val, r_val, x_test, aux_val, aux_test, args.seed)
    val_metrics = {k: base.metric(v[0], r_val) for k, v in candidates.items()}
    test_metrics = {k: base.metric(v[1], r_test) for k, v in candidates.items()}

    selected, neg, parr_val, parr_test = shared.parr_rank_from_candidates(
        candidates, parr_names, val_metrics, r_val, r_test, args.frac
    )
    candidates["PARR_rank"] = (parr_val, parr_test)
    val_metrics["PARR_rank"] = base.metric(parr_val, r_val)
    test_metrics["PARR_rank"] = base.metric(parr_test, r_test)

    plus_neg, plus_val, plus_test = shared.parr_plus_from_candidates(
        {k: v for k, v in candidates.items() if k != "PARR_rank"},
        {k: v for k, v in val_metrics.items() if k != "PARR_rank"},
    )
    candidates["PARR_plus_strong_rank"] = (plus_val, plus_test)
    val_metrics["PARR_plus_strong_rank"] = base.metric(plus_val, r_val)
    test_metrics["PARR_plus_strong_rank"] = base.metric(plus_test, r_test)

    rng = np.random.default_rng(args.seed)
    candidates["random"] = (rng.normal(size=len(r_val)), rng.normal(size=len(r_test)))
    val_metrics["random"] = np.nan
    test_metrics["random"] = base.metric(candidates["random"][1], r_test)

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
    best_epoch = int(min(history, key=lambda h: h["val_loss"])["epoch"])
    rows = []
    for scorer in keep:
        if scorer not in candidates:
            continue
        score = candidates[scorer][1]
        rows.append(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "backbone": "PatchTST",
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "d_model": args.d_model,
                "e_layers": args.e_layers,
                "n_heads": args.n_heads,
                "patch_len": args.patch_len,
                "stride": args.stride,
                "backbone_val_mse": float(r_val.mean()),
                "backbone_test_mse": float(r_test.mean()),
                "best_epoch": best_epoch,
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
    parser.add_argument("--dataset", choices=["Weather", "Exchange"], default="Weather")
    parser.add_argument("--outdir", default="local_experiments/results_boundary_patchtst_retrain")
    parser.add_argument("--max-windows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--frac", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows = evaluate(args)
    elapsed = time.perf_counter() - t0
    for row in rows:
        row["elapsed_sec"] = elapsed
    df = pd.DataFrame(rows)
    stem = f"{args.dataset.lower()}_patchtst_seed{args.seed}"
    df.to_csv(outdir / f"{stem}_scores.csv", index=False)
    summary = df.sort_values(["top25_reduction", "risk_coverage_auc"], ascending=False)
    summary.to_csv(outdir / f"{stem}_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
