import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import risk_scoring_baseline_local as base


def parr_scores(candidates, val_metrics, names):
    selected = min(names, key=lambda k: val_metrics[k])
    neg = [k for k in names if val_metrics[k] < 0] or [selected]
    weights = np.array([-val_metrics[k] for k in neg], dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    val = weights @ np.vstack([base.rank01(candidates[k][0]) for k in neg])
    test = weights @ np.vstack([base.rank01(candidates[k][1]) for k in neg])
    return val, test, selected, neg


def evaluate_case(name, rel_path, seq_len, pred_len, patch_len, args, seed):
    arr = base.numeric_frame(Path(args.root) / rel_path)
    x, y = base.make_windows(arr, seq_len, pred_len, args.max_windows, seed=seed)
    residual = base.naive_residual(x, y)
    features = base.parr_components(x, patch_len)
    aux = base.auxiliary_window_features(x)
    _, val_idx, test_idx = base.split_temporal(len(residual))

    x_val, r_val = features[val_idx], residual[val_idx]
    x_test, r_test = features[test_idx], residual[test_idx]
    aux_val, aux_test = aux[val_idx], aux[test_idx]
    candidates, parr_names = base.fit_scores(x_val, r_val, x_test, aux_val, aux_test, seed)
    val_metrics = {k: base.metric(v[0], r_val) for k, v in candidates.items()}

    parr_val, parr_test, selected, neg = parr_scores(candidates, val_metrics, parr_names)
    reduction = base.top_reduction(parr_test, r_test, args.frac)
    auc = base.coverage_auc(parr_test, r_test)

    selected_val, selected_test = candidates[selected]
    delta = abs(selected_test.mean() - selected_val.mean()) / (selected_val.std() + 1e-8)
    n_val = len(r_val)
    support_ok = n_val >= args.n_min
    drift_ok = delta <= args.gamma

    return {
        "dataset": name,
        "seed": seed,
        "n_val": n_val,
        "selected": selected,
        "neg_count": len(neg),
        "delta_score": float(delta),
        "support_ok": support_ok,
        "drift_ok": drift_ok,
        "gate_ok": support_ok and drift_ok,
        "top25_reduction": reduction,
        "risk_coverage_auc": auc,
    }


def summarize_policy(df, policy, costs):
    if policy == "ungated":
        accept = np.ones(len(df), dtype=bool)
    elif policy == "support":
        accept = df["support_ok"].to_numpy(dtype=bool)
    elif policy == "drift":
        accept = df["drift_ok"].to_numpy(dtype=bool)
    elif policy == "support+drift":
        accept = df["gate_ok"].to_numpy(dtype=bool)
    elif policy == "oracle":
        accept = df["top25_reduction"].to_numpy() > 0
    else:
        raise ValueError(policy)

    rows = []
    reductions = df["top25_reduction"].to_numpy()
    for cost in costs:
        utility = np.where(accept, reductions, -cost)
        rows.append(
            {
                "policy": policy,
                "abstain_cost": cost,
                "accept_cases": int(accept.sum()),
                "abstain_cases": int((~accept).sum()),
                "accepted_mean_reduction": float(reductions[accept].mean()) if accept.any() else np.nan,
                "negative_accepted": int(((reductions < 0) & accept).sum()),
                "mean_utility": float(utility.mean()),
                "worst_utility": float(utility.min()),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--outdir", default="local_experiments/results_multiseed")
    parser.add_argument("--max-windows", type=int, default=5000)
    parser.add_argument("--frac", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.60)
    parser.add_argument("--n-min", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 23])
    parser.add_argument("--costs", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10])
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        for name, spec in base.DATASETS.items():
            path = Path(args.root) / spec[0]
            if not path.exists():
                print(f"skip {name}: missing {path}")
                continue
            print(f"seed {seed}: gate utility {name} ...")
            rows.append(evaluate_case(name, *spec, args, seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cases = pd.DataFrame(rows)
    cases.to_csv(outdir / "gate_utility_cases.csv", index=False)

    policies = ["ungated", "support", "drift", "support+drift", "oracle"]
    summary_rows = []
    for policy in policies:
        summary_rows.extend(summarize_policy(cases, policy, args.costs))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "gate_utility_summary.csv", index=False)

    pivot = summary.pivot_table(
        index="policy",
        columns="abstain_cost",
        values="mean_utility",
        aggfunc="first",
    )
    md = ["# Local Proxy Gate Utility", ""]
    md.append(f"gamma={args.gamma}, n_min={args.n_min}, seeds={args.seeds}.")
    md.append("")
    md.append("## Policy Summary")
    md.append("")
    md.append(summary.to_markdown(index=False, floatfmt=".4f"))
    md.append("")
    md.append("## Mean Utility by Abstention Cost")
    md.append("")
    md.append(pivot.to_markdown(floatfmt=".4f"))
    (outdir / "gate_utility_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
