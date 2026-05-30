import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import risk_scoring_baseline_local as base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="local_experiments/results_multiseed")
    parser.add_argument("--max-windows", type=int, default=5000)
    parser.add_argument("--frac", type=float, default=0.25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 23])
    args = parser.parse_args()

    all_rows = []
    for seed in args.seeds:
        seed_args = SimpleNamespace(
            root=args.root,
            max_windows=args.max_windows,
            frac=args.frac,
            seed=seed,
        )
        for name, spec in base.DATASETS.items():
            path = Path(args.root) / spec[0]
            if not path.exists():
                print(f"skip {name}: missing {path}")
                continue
            print(f"seed {seed}: running {name} ...")
            rows = base.evaluate_dataset(name, *spec, seed_args)
            for row in rows:
                row["seed"] = seed
            all_rows.extend(rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(outdir / "multiseed_risk_scoring_baselines.csv", index=False)

    per_dataset = (
        df.groupby(["scorer", "dataset"], as_index=False)
        .agg(
            top25_mean=("top25_reduction", "mean"),
            top25_std=("top25_reduction", "std"),
            auc_mean=("risk_coverage_auc", "mean"),
            auc_std=("risk_coverage_auc", "std"),
            positive_rate=("top25_reduction", lambda x: (x > 0).mean()),
        )
    )
    per_dataset.to_csv(outdir / "multiseed_per_dataset.csv", index=False)

    summary = (
        per_dataset.groupby("scorer")
        .agg(
            mean_top25=("top25_mean", "mean"),
            mean_seed_std=("top25_std", "mean"),
            min_top25=("top25_mean", "min"),
            mean_auc=("auc_mean", "mean"),
            min_auc=("auc_mean", "min"),
            positive_cases=("top25_mean", lambda x: (x > 0).sum()),
            num_cases=("top25_mean", "count"),
        )
        .sort_values(["mean_top25", "mean_auc"], ascending=False)
    )
    summary.to_csv(outdir / "multiseed_summary.csv")

    md = ["# Multi-Seed Local Risk-Scoring Baseline Experiment", ""]
    md.append(f"Seeds: {', '.join(str(s) for s in args.seeds)}.")
    md.append("Proxy target: naive persistence forecasting residual on public CSV windows.")
    md.append("")
    md.append(summary.to_markdown(floatfmt=".4f"))
    md.append("")
    md.append("## Per-Dataset Mean Top-25 Reduction")
    md.append("")
    pivot = per_dataset.pivot(index="scorer", columns="dataset", values="top25_mean")
    md.append(pivot.to_markdown(floatfmt=".4f"))
    (outdir / "multiseed_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(summary.head(15))


if __name__ == "__main__":
    main()
