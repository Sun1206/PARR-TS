import argparse
import time
from pathlib import Path

import pandas as pd

import risk_scoring_baseline_local as base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="local_experiments/results_multiseed/local_overhead.csv")
    parser.add_argument("--max-windows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = []
    for name, spec in base.DATASETS.items():
        path = Path(args.root) / spec[0]
        if not path.exists():
            continue
        t0 = time.perf_counter()
        arr = base.numeric_frame(path)
        t1 = time.perf_counter()
        x, y = base.make_windows(arr, spec[1], spec[2], args.max_windows, args.seed)
        residual = base.naive_residual(x, y)
        t2 = time.perf_counter()
        features = base.parr_components(x, spec[3])
        aux = base.auxiliary_window_features(x)
        t3 = time.perf_counter()
        _, val_idx, test_idx = base.split_temporal(len(residual))
        candidates, _ = base.fit_scores(
            features[val_idx],
            residual[val_idx],
            features[test_idx],
            aux[val_idx],
            aux[test_idx],
            args.seed,
        )
        t4 = time.perf_counter()
        rows.append(
            {
                "dataset": name,
                "channels": arr.shape[1],
                "windows": len(residual),
                "load_sec": t1 - t0,
                "window_residual_sec": t2 - t1,
                "parr_feature_sec": t3 - t2,
                "strong_scorer_fit_eval_sec": t4 - t3,
                "scorers": len(candidates),
            }
        )
        print(rows[-1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
