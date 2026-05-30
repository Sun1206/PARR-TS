from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = (
    ROOT
    / "remote_results"
    / "risk_coverage_neural_applied_20260516_142149"
    / "risk_coverage_gate_applied_summary.csv"
)
OUT_DIR = Path(__file__).resolve().parent

METHOD_ORDER = ["PARR-rank", "GBDT", "Ridge", "Mahalanobis OOD", "MLP"]
STYLE = {
    "PARR-rank": ("#1f77b4", "o"),
    "GBDT": ("#2ca02c", "s"),
    "Ridge": ("#9467bd", "^"),
    "Mahalanobis OOD": ("#8c564b", "D"),
    "MLP": ("#d62728", "v"),
}


def load_rows() -> dict[str, list[tuple[float, float]]]:
    rows: dict[str, list[tuple[float, float]]] = {m: [] for m in METHOD_ORDER}
    with SUMMARY.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            if method not in rows:
                continue
            coverage = 100.0 * float(row["coverage"])
            reduction = float(row["mean_pct"])
            rows[method].append((coverage, reduction))
    for method in rows:
        rows[method].sort()
    return rows


def main() -> None:
    rows = load_rows()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 5.4,
            "axes.labelsize": 5.6,
            "xtick.labelsize": 4.9,
            "ytick.labelsize": 4.9,
            "legend.fontsize": 4.5,
            "axes.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.2, 1.9))
    for method in METHOD_ORDER:
        points = rows[method]
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        color, marker = STYLE[method]
        ax.plot(
            x,
            y,
            label=method,
            color=color,
            marker=marker,
            markersize=2.35,
            linewidth=0.72,
            markeredgewidth=0.45,
        )

    ax.axhline(0, color="black", linewidth=0.55, alpha=0.55)
    ax.set_xlabel("Coverage (%)", labelpad=1.5)
    ax.set_ylabel("Mean MSE reduction (%)", labelpad=2.0)
    ax.set_xticks([10, 25, 50, 75])
    ax.set_xlim(7, 78)
    ax.set_ylim(-5, 22)
    ax.grid(axis="y", linewidth=0.3, alpha=0.35)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper right",
        handlelength=1.25,
        columnspacing=0.8,
        handletextpad=0.35,
        borderaxespad=0.2,
    )
    fig.tight_layout(pad=0.1)
    fig.savefig(OUT_DIR / "risk_coverage_gate_applied.pdf")
    fig.savefig(OUT_DIR / "risk_coverage_gate_applied.png", dpi=300)


if __name__ == "__main__":
    main()
