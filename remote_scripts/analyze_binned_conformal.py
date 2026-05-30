import argparse
import glob
import os

import numpy as np


def find(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[-1]


def qhat(values, alpha):
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return np.nan
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(values, level, method="higher"))


def pointwise_metrics(abs_err, q):
    covered = abs_err <= q
    return {
        "coverage": float(covered.mean()),
        "width": float(2 * q),
        "q": float(q),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="results/*raw_sigmoid_score*")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--bins", type=int, default=4)
    args = parser.parse_args()

    folder = find(args.pattern)
    pred = np.load(os.path.join(folder, "pred.npy"))
    true = np.load(os.path.join(folder, "true.npy"))
    score = np.load(os.path.join(folder, "parr_score.npy"))

    n = len(score)
    split = n // 2
    cal_idx = np.arange(split)
    eval_idx = np.arange(split, n)

    abs_err = np.abs(pred - true)
    cal_err = abs_err[cal_idx].reshape(-1)
    eval_err = abs_err[eval_idx]
    eval_score = score[eval_idx]

    global_q = qhat(cal_err, args.alpha)
    global_overall = pointwise_metrics(eval_err.reshape(-1), global_q)

    # Use calibration-score quantiles to define deployment-time score bins.
    edges = np.quantile(score[cal_idx], np.linspace(0, 1, args.bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    bin_rows = []
    widths = []
    cover_counts = []
    total_counts = []
    for b in range(args.bins):
        lo, hi = edges[b], edges[b + 1]
        cal_mask = (score[cal_idx] > lo) & (score[cal_idx] <= hi)
        eval_mask = (eval_score > lo) & (eval_score <= hi)
        cal_bin_err = abs_err[cal_idx][cal_mask].reshape(-1)
        eval_bin_err = eval_err[eval_mask].reshape(-1)
        q = qhat(cal_bin_err, args.alpha)
        if np.isnan(q) or len(eval_bin_err) == 0:
            continue
        m = pointwise_metrics(eval_bin_err, q)
        global_m = pointwise_metrics(eval_bin_err, global_q)
        bin_rows.append(
            {
                "bin": b + 1,
                "eval_samples": int(eval_mask.sum()),
                "score_lo": float(np.min(eval_score[eval_mask])) if eval_mask.any() else None,
                "score_hi": float(np.max(eval_score[eval_mask])) if eval_mask.any() else None,
                "binned_coverage": m["coverage"],
                "binned_width": m["width"],
                "binned_q": m["q"],
                "global_coverage_in_bin": global_m["coverage"],
                "global_width": global_m["width"],
            }
        )
        widths.append(m["width"] * len(eval_bin_err))
        cover_counts.append((eval_bin_err <= q).sum())
        total_counts.append(len(eval_bin_err))

    total_points = np.sum(total_counts)
    binned_overall = {
        "coverage": float(np.sum(cover_counts) / total_points),
        "width": float(np.sum(widths) / total_points),
    }

    print("folder", folder)
    print("alpha", args.alpha, "cal_samples", len(cal_idx), "eval_samples", len(eval_idx))
    print("global", global_overall)
    print("binned_overall", binned_overall)
    print("width_reduction_vs_global", float(1 - binned_overall["width"] / global_overall["width"]))
    for row in bin_rows:
        print(row)


if __name__ == "__main__":
    main()

