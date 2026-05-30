import glob
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr


def main():
    rows = []
    for folder in sorted(glob.glob("results/*")):
        metrics_path = os.path.join(folder, "metrics.npy")
        if not os.path.exists(metrics_path):
            continue
        metrics = np.load(metrics_path)
        row = {
            "setting": os.path.basename(folder),
            "mae": float(metrics[0]),
            "mse": float(metrics[1]),
        }
        score_path = os.path.join(folder, "parr_score.npy")
        residual_path = os.path.join(folder, "parr_residual_mse.npy")
        if os.path.exists(score_path) and os.path.exists(residual_path):
            score = np.load(score_path)
            residual = np.load(residual_path)
            row.update(
                {
                    "score_mean": float(score.mean()),
                    "score_std": float(score.std()),
                    "spearman_score_residual": float(spearmanr(score, residual).statistic),
                    "pearson_score_residual": float(pearsonr(score, residual).statistic),
                }
            )
        rows.append(row)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

