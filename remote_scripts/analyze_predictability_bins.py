import argparse
import glob
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr


def find(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="results/*raw_sigmoid_score*")
    parser.add_argument("--bins", type=int, default=4)
    args = parser.parse_args()

    folder = find(args.pattern)
    score = np.load(os.path.join(folder, "parr_score.npy"))
    residual = np.load(os.path.join(folder, "parr_residual_mse.npy"))
    pred = np.load(os.path.join(folder, "pred.npy"))
    true = np.load(os.path.join(folder, "true.npy"))
    mae_sample = np.abs(pred - true).mean(axis=(1, 2))

    order = np.argsort(score)
    splits = np.array_split(order, args.bins)
    print("folder", folder)
    print("n", len(score))
    print("score_residual_spearman", float(spearmanr(score, residual).statistic))
    print("score_residual_pearson", float(pearsonr(score, residual).statistic))
    for i, idx in enumerate(splits, 1):
        print(
            {
                "bin": i,
                "n": int(len(idx)),
                "score_min": float(score[idx].min()),
                "score_max": float(score[idx].max()),
                "mse_mean": float(residual[idx].mean()),
                "mae_mean": float(mae_sample[idx].mean()),
                "mse_p90": float(np.quantile(residual[idx], 0.9)),
            }
        )


if __name__ == "__main__":
    main()

