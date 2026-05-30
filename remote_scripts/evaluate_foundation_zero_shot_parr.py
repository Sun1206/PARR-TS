import argparse
import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "parr_icdm"))

from evaluate_calibrated_conformal import collect_full
from evaluate_score_ensembles import candidate_scores, rank_ensemble
from evaluate_strong_risk_score_baselines import fit_strong_scores, top_reduction
from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
from layers.PARR import PARRPreprocessor


def metric(score, residual):
    value = spearmanr(score, residual).statistic
    return float(value) if not np.isnan(value) else 0.0


def coverage_auc(score, residual, coverages=(0.10, 0.25, 0.50, 0.75)):
    return float(np.mean([top_reduction(score, residual, c) for c in coverages]))


def foundation_args(model, dataset):
    common = dict(
        task_name="zero_shot_forecast",
        is_training=0,
        root_path="./dataset/ETT-small/",
        data_path=f"{dataset}.csv",
        model_id=f"{dataset}_96_96_{model.lower()}_foundation",
        model=model,
        data=dataset,
        features="M",
        target="OT",
        freq="h",
        checkpoints="./checkpoints/",
        seq_len=96,
        label_len=0,
        pred_len=96,
        seasonal_patterns="Monthly",
        inverse=False,
        mask_rate=0.25,
        anomaly_ratio=0.25,
        expand=2,
        d_conv=4,
        tv_dt=0,
        tv_B=0,
        tv_C=0,
        use_D=0,
        top_k=5,
        num_kernels=6,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=24,
        num_workers=0,
        itr=1,
        train_epochs=1,
        batch_size=16,
        patience=1,
        learning_rate=0.0001,
        des="foundation",
        loss="MSE",
        lradj="type1",
        use_amp=False,
        use_gpu=True,
        gpu=0,
        gpu_type="cuda",
        use_multi_gpu=False,
        devices="0,1,2,3",
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_dtw=False,
        augmentation_ratio=0,
        seed=2,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag="",
        parr_patch_len=16,
        parr_alpha_s=1.0,
        parr_alpha_d=1.0,
        parr_alpha_e=1.0,
        parr_alpha_g=1.0,
        parr_min_keep=1e-4,
        parr_dropout=0.0,
        parr_replace_strength=0.0,
        parr_weighted_loss=False,
        parr_save_diagnostics=False,
        parr_score_mode="sigmoid_raw",
        use_parr=False,
    )
    return SimpleNamespace(**common)


def parr_rank_score(x_val, y_val, x_test):
    candidates = candidate_scores(x_val, y_val, x_test)
    ranked = sorted(candidates, key=lambda name: candidates[name]["val_spearman"])
    selected = ranked[0]
    negative = [name for name in ranked if candidates[name]["val_spearman"] < 0] or [selected]
    weights = np.array([-candidates[name]["val_spearman"] for name in negative], dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    return rank_ensemble(candidates, negative, "test", weights), selected, ",".join(negative)


def evaluate(model, dataset, seed):
    cfg = foundation_args(model, dataset)
    exp = Exp_Zero_Shot_Forecast(cfg)
    parr = PARRPreprocessor(cfg).eval()
    x_val, y_val, _ = collect_full(exp, parr, "val")
    x_test, y_test, _ = collect_full(exp, parr, "test")
    parr_score, selected, negative_pool = parr_rank_score(x_val, y_val, x_test)
    strong = fit_strong_scores(x_val, y_val, x_test, seed)
    return {
        "model": model,
        "dataset": dataset,
        "n_val": len(y_val),
        "n_test": len(y_test),
        "overall_mse": float(y_test.mean()),
        "selected": selected,
        "negative_pool": negative_pool,
        "parr_top10": top_reduction(parr_score, y_test, 0.10),
        "parr_top25": top_reduction(parr_score, y_test, 0.25),
        "parr_top50": top_reduction(parr_score, y_test, 0.50),
        "parr_auc": coverage_auc(parr_score, y_test),
        "parr_spearman": metric(parr_score, y_test),
        "ridge_top25": top_reduction(strong["ridge_risk"][1], y_test, 0.25),
        "gbdt_top25": top_reduction(strong["gbdt_risk"][1], y_test, 0.25),
        "maha_top25": top_reduction(strong["mahalanobis_ood"][1], y_test, 0.25),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Chronos")
    parser.add_argument("--dataset", default="ETTm2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    row = evaluate(args.model, args.dataset, args.seed)
    print(row, flush=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(row))
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
