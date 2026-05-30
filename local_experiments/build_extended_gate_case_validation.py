from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "local_experiments" / "results_extended_gate"
STRONG = ROOT / "remote_results" / "strong_neural_baselines_20260516_135228" / "strong_neural_baselines_20260516_135228.csv"
TRAFFIC = ROOT / "local_experiments" / "results_traffic_patchtst_full" / "traffic_patchtst_full_summary.csv"


BACKBONE = {
    "timemixer": "TM",
    "patchtst": "PT",
    "timexer": "TX",
    "itransformer": "iTr",
}


def case_fields(case: str):
    parts = case.split("_")
    dataset = "_".join(parts[:-1])
    backbone = parts[-1]
    if dataset.startswith("ett"):
        label = {
            "etth1": "ETTh1",
            "etth2": "ETTh2",
            "ettm1": "ETTm1",
            "ettm2": "ETTm2",
        }[dataset]
        delta = "0.568" if dataset == "ettm2" else "<=0.568"
        bsign = "0.69"
        gate_before = "apply"
        final_gate = "apply"
    elif dataset == "electricity":
        label = "Elec."
        delta = {
            "timemixer": "0.536",
            "patchtst": "0.363",
            "timexer": "0.572",
            "itransformer": "0.357",
        }[backbone]
        bsign = "0.88"
        gate_before = "apply"
        final_gate = "apply"
    elif dataset == "weather":
        label = "Weather"
        delta = "0.952"
        bsign = "0.92"
        gate_before = "abstain"
        final_gate = "abstain"
    elif dataset == "exchange":
        label = "Exchange"
        delta = ">=7.57"
        bsign = "0.47"
        gate_before = "abstain"
        final_gate = "abstain"
    elif dataset == "illness":
        label = "Illness"
        delta = "<=0.462"
        bsign = "0.96"
        gate_before = "abstain"
        final_gate = "abstain"
    else:
        raise ValueError(case)
    return f"{label}/{BACKBONE[backbone]}", delta, bsign, gate_before, final_gate


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    strong = pd.read_csv(STRONG)
    rows = []
    for _, row in strong.iterrows():
        case, delta, bsign, gate_before, final_gate = case_fields(row["case"])
        rows.append(
            {
                "case": case,
                "n_val": int(row["n_val"]),
                "Delta_score": delta,
                "B_sign": bsign,
                "gate before sign": gate_before,
                "final gate": final_gate,
                "PARR Top-25": f"{100.0 * row['PARR_rank_top25']:.1f}",
            }
        )

    traffic = pd.read_csv(TRAFFIC)
    parr = traffic.loc[traffic["scorer"] == "PARR_rank"].iloc[0]
    rows.append(
        {
            "case": "Traffic/PT-full",
            "n_val": int(parr["n_val"]),
            "Delta_score": "0.13",
            "B_sign": "0.60",
            "gate before sign": "apply",
            "final gate": "abstain",
            "PARR Top-25": f"{100.0 * parr['top25_reduction']:.1f}",
        }
    )

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "extended_gate_case_validation.csv", index=False)

    md = [
        "## Extended Gate Case-Level Validation",
        "",
        "`B_sign` is computed from chronological validation blocks before held-out test residuals are inspected. "
        "`PARR Top-25` is reported only after the gate as an audit outcome.",
        "",
        df.to_markdown(index=False),
        "",
    ]
    (OUTDIR / "extended_gate_case_validation.md").write_text("\n".join(md), encoding="utf-8")

    def latex_delta(value: str) -> str:
        if value.startswith("<="):
            return "$\\le" + value[2:] + "$"
        if value.startswith(">="):
            return "$\\ge" + value[2:] + "$"
        return value

    latex_rows = []
    for _, r in df.iterrows():
        latex_rows.append(
            f"{r['case']} & {r['n_val']} & {latex_delta(r['Delta_score'])} & {r['B_sign']} & "
            f"{r['gate before sign']} & {r['final gate']} & {r['PARR Top-25']} \\\\"
        )
    (OUTDIR / "extended_gate_case_validation_rows.tex").write_text("\n".join(latex_rows) + "\n", encoding="utf-8")
    print(df)


if __name__ == "__main__":
    main()
