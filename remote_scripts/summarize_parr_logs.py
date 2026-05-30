import ast
import csv
import json
import os
from pathlib import Path


ROOT = Path("/root/autodl-tmp/parr_ts_icdm/Time-Series-Library")
LOGS = ROOT / "logs"
OUT = ROOT / "logs" / "parr_summary"


def iter_dict_rows(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            row = ast.literal_eval(line)
        except Exception:
            continue
        if isinstance(row, dict):
            yield row


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for key, value in row.items():
                if isinstance(value, (list, tuple, dict)):
                    cleaned[key] = json.dumps(value, ensure_ascii=False)
                else:
                    cleaned[key] = value
            writer.writerow(cleaned)


def pct(x):
    return None if x is None else 100.0 * float(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    horizon_rows = []
    for name in ["horizon_robustness_etth1_timemixer.log", "horizon_robustness_etth2_timemixer.log", "horizon_robustness_ettm1_timemixer.log"]:
        for row in iter_dict_rows(LOGS / name):
            horizon_rows.append(row)
    write_csv(
        OUT / "horizon_robustness.csv",
        horizon_rows,
        [
            "dataset",
            "model",
            "horizon",
            "overall_mse",
            "hard_candidate",
            "hard_selected_top25_reduction",
            "top3_rank_top25_reduction",
            "negative_weighted_rank_top25_reduction",
            "oracle_candidate",
            "oracle_candidate_top25_reduction",
            "hard_selected_spearman",
            "top3_rank_spearman",
            "negative_weighted_rank_spearman",
            "oracle_candidate_spearman",
        ],
    )

    selective_rows = list(iter_dict_rows(LOGS / "selective_risk_candidate_pool_oracle_gap.log"))
    write_csv(OUT / "selective_risk_hard_oracle.csv", selective_rows)

    boot_rows = list(iter_dict_rows(LOGS / "selective_risk_bootstrap_500.log"))
    write_csv(OUT / "selective_risk_bootstrap.csv", boot_rows)

    horizon_boot_rows = list(iter_dict_rows(LOGS / "horizon_bootstrap_ci.log"))
    write_csv(
        OUT / "horizon_bootstrap_ci.csv",
        horizon_boot_rows,
        [
            "dataset",
            "model",
            "horizon",
            "overall_mse",
            "hard_candidate",
            "hard_selected_top25_reduction",
            "hard_selected_ci95",
            "negative_weighted_rank_top25_reduction",
            "negative_weighted_rank_ci95",
            "oracle_candidate",
            "oracle_candidate_top25_reduction",
            "oracle_candidate_ci95",
        ],
    )

    # Compact Markdown snippets for direct paper insertion.
    md = []
    if horizon_rows:
        md.append("# Horizon Robustness")
        md.append("")
        md.append("| Dataset | Horizon | Overall MSE | Hard scorer | Hard Top25 | PARR-rank Top25 | Oracle Top25 |")
        md.append("|---|---:|---:|---|---:|---:|---:|")
        for row in sorted(horizon_rows, key=lambda r: (r.get("dataset", ""), int(r.get("horizon", 0)))):
            rank_value = row.get("negative_weighted_rank_top25_reduction")
            if rank_value is None:
                rank_value = row.get("top3_rank_top25_reduction")
            md.append(
                "| {dataset} | {horizon} | {overall:.4f} | {hard} | {hard_pct:.1f}% | {rank_pct:.1f}% | {oracle_pct:.1f}% |".format(
                    dataset=row.get("dataset"),
                    horizon=row.get("horizon"),
                    overall=float(row.get("overall_mse")),
                    hard=row.get("hard_candidate"),
                    hard_pct=pct(row.get("hard_selected_top25_reduction")),
                    rank_pct=pct(rank_value),
                    oracle_pct=pct(row.get("oracle_candidate_top25_reduction")),
                )
            )
    if horizon_boot_rows:
        md.append("")
        md.append("# Horizon Bootstrap CI")
        md.append("")
        md.append("| Dataset | Horizon | Hard Top25 95% CI | PARR-rank Top25 95% CI |")
        md.append("|---|---:|---:|---:|")
        for row in sorted(horizon_boot_rows, key=lambda r: (r.get("dataset", ""), int(r.get("horizon", 0)))):
            hard_ci = row.get("hard_selected_ci95")
            rank_ci = row.get("negative_weighted_rank_ci95")
            md.append(
                "| {dataset} | {horizon} | [{hard_lo:.1f}%, {hard_hi:.1f}%] | [{rank_lo:.1f}%, {rank_hi:.1f}%] |".format(
                    dataset=row.get("dataset"),
                    horizon=row.get("horizon"),
                    hard_lo=pct(hard_ci[0]),
                    hard_hi=pct(hard_ci[1]),
                    rank_lo=pct(rank_ci[0]),
                    rank_hi=pct(rank_ci[1]),
                )
            )
    (OUT / "paper_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    summary = {
        "horizon_rows": len(horizon_rows),
        "selective_rows": len(selective_rows),
        "bootstrap_rows": len(boot_rows),
        "horizon_bootstrap_rows": len(horizon_boot_rows),
        "output_dir": str(OUT),
    }
    print(summary)


if __name__ == "__main__":
    main()
