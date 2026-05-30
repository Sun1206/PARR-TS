import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import frozen_shared_linear_strong_baselines as shared
import risk_scoring_baseline_local as base


class TrafficWindowDataset(Dataset):
    def __init__(self, arr, indices, seq_len, pred_len, mean, std):
        self.arr = arr
        self.indices = np.asarray(indices, dtype=np.int64)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = int(self.indices[item])
        x = self.arr[i : i + self.seq_len]
        y = self.arr[i + self.seq_len : i + self.seq_len + self.pred_len]
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


class PatchTSTForecaster(nn.Module):
    def __init__(self, root, seq_len, pred_len, enc_in, args):
        super().__init__()
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


def split_indices(n):
    a = int(n * 0.7)
    b = int(n * 0.8)
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)


def train_window_stats(arr, train_idx, seq_len, chunk=128):
    total = np.zeros(arr.shape[1], dtype=np.float64)
    total_sq = np.zeros(arr.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(train_idx), chunk):
        idx = train_idx[start : start + chunk]
        windows = np.stack([arr[i : i + seq_len] for i in idx]).astype(np.float64)
        total += windows.sum(axis=(0, 1))
        total_sq += (windows * windows).sum(axis=(0, 1))
        count += windows.shape[0] * windows.shape[1]
    mean = total / count
    var = np.maximum(total_sq / count - mean * mean, 1e-12)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32) + 1e-6


def residual_mse(pred, y):
    return ((pred - y) ** 2).mean(axis=(1, 2))


def train_model(arr, train_idx, val_idx, mean, std, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = PatchTSTForecaster(args.root, args.seq_len, args.pred_len, arr.shape[1], args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    train_ds = TrafficWindowDataset(arr, train_idx, args.seq_len, args.pred_len, mean, std)
    val_ds = TrafficWindowDataset(arr, val_idx, args.seq_len, args.pred_len, mean, std)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    best_state = None
    best_val = float("inf")
    patience_left = args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        epoch_t0 = time.perf_counter()
        for step, (xb, yb) in enumerate(train_loader, 1):
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
            train_losses.append(float(loss.detach().cpu()))
            if args.log_every and step % args.log_every == 0:
                print(
                    f"epoch {epoch:02d} step {step}/{len(train_loader)} "
                    f"loss={np.mean(train_losses[-args.log_every:]):.6f}",
                    flush=True,
                )

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    val_losses.append(float(loss_fn(model(xb), yb).detach().cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        elapsed = time.perf_counter() - epoch_t0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_sec": elapsed,
            }
        )
        print(
            f"epoch {epoch:02d} train={train_loss:.6f} val={val_loss:.6f} sec={elapsed:.1f}",
            flush=True,
        )
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
    return model, device, pd.DataFrame(history)


def collect_residuals_and_features(model, device, arr, indices, mean, std, args):
    ds = TrafficWindowDataset(arr, indices, args.seq_len, args.pred_len, mean, std)
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    residuals = []
    parr = []
    aux = []
    loss_fn = nn.MSELoss(reduction="none")
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred = model(xb)
                batch_resid = loss_fn(pred, yb).mean(dim=(1, 2)).detach().cpu().float().numpy()
            x_np = xb.detach().cpu().float().numpy()
            residuals.append(batch_resid)
            parr.append(base.parr_components(x_np, args.patch_len))
            aux.append(base.auxiliary_window_features(x_np))
    return np.concatenate(residuals), np.vstack(parr), np.vstack(aux)


def evaluate_scores(r_val, r_test, x_val, x_test, aux_val, aux_test, args):
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
    rows = []
    for scorer in keep:
        score = candidates[scorer][1]
        rows.append(
            {
                "dataset": "Traffic",
                "seed": args.seed,
                "backbone": args.backbone_name,
                "n_train": args.n_train,
                "n_val": len(r_val),
                "n_test": len(r_test),
                "d_model": args.d_model,
                "d_ff": args.d_ff,
                "e_layers": args.e_layers,
                "n_heads": args.n_heads,
                "patch_len": args.patch_len,
                "stride": args.stride,
                "backbone_val_mse": float(r_val.mean()),
                "backbone_test_mse": float(r_test.mean()),
                "best_epoch": args.best_epoch,
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
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="local_experiments/results_traffic_patchtst_full")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--frac", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--backbone-name", default="PatchTST-full")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    arr = base.numeric_frame(Path(args.root) / "data_cache/traffic.csv")
    n_windows = len(arr) - args.seq_len - args.pred_len + 1
    train_idx, val_idx, test_idx = split_indices(n_windows)
    print(
        f"Traffic full windows={n_windows} train={len(train_idx)} "
        f"val={len(val_idx)} test={len(test_idx)} channels={arr.shape[1]}",
        flush=True,
    )
    mean, std = train_window_stats(arr, train_idx, args.seq_len)
    model, device, history = train_model(arr, train_idx, val_idx, mean, std, args)
    history.to_csv(outdir / "traffic_patchtst_full_history.csv", index=False)
    args.best_epoch = int(history.sort_values("val_loss").iloc[0]["epoch"])
    args.n_train = len(train_idx)

    r_val, x_val, aux_val = collect_residuals_and_features(model, device, arr, val_idx, mean, std, args)
    r_test, x_test, aux_test = collect_residuals_and_features(model, device, arr, test_idx, mean, std, args)
    np.savez_compressed(
        outdir / "traffic_patchtst_full_residual_features.npz",
        r_val=r_val,
        r_test=r_test,
        x_val=x_val,
        x_test=x_test,
        aux_val=aux_val,
        aux_test=aux_test,
    )

    df = evaluate_scores(r_val, r_test, x_val, x_test, aux_val, aux_test, args)
    df["elapsed_sec"] = time.perf_counter() - t0
    df.to_csv(outdir / "traffic_patchtst_full_scores.csv", index=False)
    summary = df.sort_values(["top25_reduction", "risk_coverage_auc"], ascending=False)
    summary.to_csv(outdir / "traffic_patchtst_full_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
