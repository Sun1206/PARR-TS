import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "parr_icdm"))

from evaluate_calibrated_conformal import collect_full, fit_scores_for_val_test
from evaluate_score_ensembles import candidate_scores, rank_ensemble
from evaluate_val_calibrated_parr import base_args
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from layers.PARR import PARRPreprocessor


DEFAULT_CASES = [
    "etth1_timemixer",
    "etth1_patchtst",
    "etth1_timexer",
    "etth1_itransformer",
    "etth2_timemixer",
    "etth2_patchtst",
    "etth2_timexer",
    "etth2_itransformer",
    "ettm1_timemixer",
    "ettm1_patchtst",
    "ettm1_timexer",
    "ettm1_itransformer",
    "ettm2_timemixer",
    "ettm2_patchtst",
    "ettm2_timexer",
    "ettm2_itransformer",
    "electricity_timemixer",
    "electricity_patchtst",
    "electricity_timexer",
    "electricity_itransformer",
    "weather_timemixer",
    "weather_patchtst",
    "weather_timexer",
    "weather_itransformer",
    "exchange_timemixer",
    "exchange_patchtst",
    "exchange_timexer",
    "exchange_itransformer",
    "illness_timemixer",
    "illness_patchtst",
    "illness_timexer",
    "illness_itransformer",
]


def metric(score, residual):
    value = spearmanr(score, residual).statistic
    return float(value) if not np.isnan(value) else 0.0


def rank01(score):
    if len(score) <= 1:
        return np.zeros_like(score, dtype=np.float64)
    return (rankdata(score, method="average") - 1.0) / (len(score) - 1.0)


def top_reduction(score, residual, frac=0.25):
    order = np.argsort(score)
    k = max(1, int(frac * len(order)))
    selected = float(residual[order[-k:]].mean())
    overall = float(residual.mean())
    return float(1.0 - selected / overall)


def coverage_auc(score, residual, coverages=(0.1, 0.25, 0.5, 0.75)):
    return float(np.mean([top_reduction(score, residual, c) for c in coverages]))


def fit_strong_scores(x_val, y_val, x_test, seed):
    scaler = StandardScaler().fit(x_val)
    zv = scaler.transform(x_val)
    zt = scaler.transform(x_test)
    scores = {}

    ridge = Ridge(alpha=1.0).fit(zv, y_val)
    scores["ridge_risk"] = (-ridge.predict(zv), -ridge.predict(zt))

    gbdt = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.04,
        max_leaf_nodes=8,
        l2_regularization=0.05,
        random_state=seed,
    ).fit(zv, y_val)
    scores["gbdt_risk"] = (-gbdt.predict(zv), -gbdt.predict(zt))

    mlp = MLPRegressor(
        hidden_layer_sizes=(24,),
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=800,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=seed,
    ).fit(zv, y_val)
    scores["mlp_risk"] = (-mlp.predict(zv), -mlp.predict(zt))

    k = min(50, len(zv))
    nn = NearestNeighbors(n_neighbors=k).fit(zv)
    _, val_nn = nn.kneighbors(zv)
    _, test_nn = nn.kneighbors(zt)
    scores["knn_mean_risk"] = (-y_val[val_nn].mean(axis=1), -y_val[test_nn].mean(axis=1))
    scores["knn_q90_risk"] = (
        -np.quantile(y_val[val_nn], 0.9, axis=1),
        -np.quantile(y_val[test_nn], 0.9, axis=1),
    )

    lw = LedoitWolf().fit(zv)
    scores["mahalanobis_ood"] = (-lw.mahalanobis(zv), -lw.mahalanobis(zt))

    n_comp = min(4, zv.shape[1], max(1, len(zv) - 1))
    pca = PCA(n_components=n_comp, random_state=seed).fit(zv)
    val_rec = pca.inverse_transform(pca.transform(zv))
    test_rec = pca.inverse_transform(pca.transform(zt))
    scores["pca_reconstruction_ood"] = (
        -((zv - val_rec) ** 2).mean(axis=1),
        -((zt - test_rec) ** 2).mean(axis=1),
    )
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-jsonl", default="")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    rows = []
    jsonl_handle = None
    if args.out_jsonl:
        os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
        jsonl_handle = open(args.out_jsonl, "w", encoding="utf-8")

    for case in args.cases:
        cfg = base_args(case)
        ckpts = sorted(glob.glob(cfg.ckpt_glob))
        if not ckpts:
            row = {"case": case, "error": "checkpoint_not_found", "glob": cfg.ckpt_glob}
            print(row, flush=True)
            rows.append(row)
            if jsonl_handle:
                jsonl_handle.write(json.dumps(row) + "\n")
                jsonl_handle.flush()
            continue

        exp = Exp_Long_Term_Forecast(cfg)
        exp.model.load_state_dict(torch.load(ckpts[-1], map_location=exp.device))
        parr = PARRPreprocessor(cfg).eval()
        x_val, y_val, _ = collect_full(exp, parr, "val")
        x_test, y_test, _ = collect_full(exp, parr, "test")

        parr_candidates = candidate_scores(x_val, y_val, x_test)
        ranked = sorted(parr_candidates, key=lambda name: parr_candidates[name]["val_spearman"])
        selected = ranked[0]
        negative = [name for name in ranked if parr_candidates[name]["val_spearman"] < 0] or [selected]
        weights = np.array([-parr_candidates[name]["val_spearman"] for name in negative], dtype=np.float64)
        weights = weights / (weights.sum() + 1e-12)
        parr_rank = rank_ensemble(parr_candidates, negative, "test", weights)

        strong = fit_strong_scores(x_val, y_val, x_test, args.seed)
        all_val_scores = {name: item["val"] for name, item in parr_candidates.items()}
        all_test_scores = {name: item["test"] for name, item in parr_candidates.items()}
        for name, (val_score, test_score) in strong.items():
            all_val_scores[name] = val_score
            all_test_scores[name] = test_score
        neg = [name for name, val_score in all_val_scores.items() if metric(val_score, y_val) < 0]
        if not neg:
            neg = [selected]
        strong_weights = np.array([-metric(all_val_scores[name], y_val) for name in neg], dtype=np.float64)
        strong_weights = strong_weights / (strong_weights.sum() + 1e-12)
        strong_rank = np.vstack([rank01(all_test_scores[name]) for name in neg])
        parr_plus = strong_weights @ strong_rank

        row = {"case": case, "n_val": len(y_val), "n_test": len(y_test)}
        methods = {
            "PARR_rank": parr_rank,
            "PARR_plus_strong_rank": parr_plus,
        }
        for name, (_, test_score) in strong.items():
            methods[name] = test_score
        for name, score in methods.items():
            row[f"{name}_spearman"] = metric(score, y_test)
            row[f"{name}_top25"] = top_reduction(score, y_test)
            row[f"{name}_auc"] = coverage_auc(score, y_test)
        print(row, flush=True)
        rows.append(row)
        if jsonl_handle:
            jsonl_handle.write(json.dumps(row) + "\n")
            jsonl_handle.flush()

    if jsonl_handle:
        jsonl_handle.close()

    if args.out_csv and rows:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        keys = sorted({key for row in rows for key in row})
        with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
