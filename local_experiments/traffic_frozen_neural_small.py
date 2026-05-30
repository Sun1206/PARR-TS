import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import frozen_shared_linear_strong_baselines as shared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="local_experiments/results_traffic_neural_small")
    parser.add_argument("--max-windows", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--frac", type=float, default=0.25)
    args = parser.parse_args()

    run_args = SimpleNamespace(
        root=args.root,
        max_windows=args.max_windows,
        frac=args.frac,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        cpu=args.cpu,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    rows = shared.evaluate_dataset(
        "Traffic",
        "data_cache/traffic.csv",
        96,
        96,
        16,
        run_args,
        args.seed,
    )
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(rows)
    df["elapsed_sec"] = elapsed
    df.to_csv(outdir / "traffic_frozen_neural_small_scores.csv", index=False)

    summary = (
        df.sort_values(["top25_reduction", "risk_coverage_auc"], ascending=False)
        [
            [
                "dataset",
                "backbone",
                "seed",
                "n_train",
                "n_val",
                "n_test",
                "backbone_val_mse",
                "backbone_test_mse",
                "best_epoch",
                "scorer",
                "val_spearman",
                "test_spearman",
                "top25_reduction",
                "risk_coverage_auc",
                "elapsed_sec",
                "parr_neg_pool",
            ]
        ]
    )
    summary.to_csv(outdir / "traffic_frozen_neural_small_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
