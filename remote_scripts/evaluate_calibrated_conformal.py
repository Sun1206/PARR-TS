import argparse
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "parr_icdm"))

from evaluate_val_calibrated_parr import base_args, sigmoid
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from layers.PARR import PARRPreprocessor


def qhat(values, alpha):
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return np.nan
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(values, level, method="higher"))


def point_metrics(abs_err, q):
    return {
        "coverage": float((abs_err <= q).mean()),
        "width": float(2.0 * q),
        "q": float(q),
    }


def collect_full(exp, parr, flag):
    _, loader = exp._get_data(flag=flag)
    comps, preds, trues = [], [], []
    exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            x_cpu = batch_x.float()
            patches, _ = parr._patchify(x_cpu)
            comp = torch.stack(
                [
                    parr._spectral_entropy(patches).mean(dim=1),
                    parr._period_drift(patches).mean(dim=1),
                    parr._smooth_residual(patches).mean(dim=1),
                    parr._channel_profile_drift(patches).mean(dim=1),
                ],
                dim=1,
            ).cpu().numpy()

            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len :, :]).float()
            dec_inp = torch.cat([batch_y[:, : exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)
            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if exp.args.features == "MS" else 0
            outputs = outputs[:, -exp.args.pred_len :, f_dim:]
            truth = batch_y[:, -exp.args.pred_len :, f_dim:]
            comps.append(comp)
            preds.append(outputs.detach().cpu().numpy())
            trues.append(truth.detach().cpu().numpy())

    x = np.concatenate(comps, axis=0)
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    abs_err = np.abs(pred - true)
    residual = ((pred - true) ** 2).mean(axis=(1, 2))
    return x, residual, abs_err


def fit_scores_for_val_test(x_val, y_val, x_test):
    mu = x_val.mean(axis=0)
    sigma = x_val.std(axis=0) + 1e-8
    z_val = (x_val - mu) / sigma
    z_test = (x_test - mu) / sigma

    signs = []
    for i in range(z_val.shape[1]):
        s = spearmanr(z_val[:, i], y_val).statistic
        signs.append(0.0 if np.isnan(s) or abs(s) < 0.03 else float(np.sign(s)))
    signs = np.asarray(signs)
    sign_val = sigmoid(-(z_val @ signs))
    sign_test = sigmoid(-(z_test @ signs))

    x1_val = np.c_[np.ones(len(z_val)), z_val]
    x1_test = np.c_[np.ones(len(z_test)), z_test]
    ridge = 1e-3 * np.eye(x1_val.shape[1])
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x1_val.T @ x1_val + ridge, x1_val.T @ y_val)
    ridge_val = -(x1_val @ beta)
    ridge_test = -(x1_test @ beta)

    out = {
        "sign": {"val": sign_val, "test": sign_test, "beta": signs.tolist()},
        "ridge": {"val": ridge_val, "test": ridge_test, "beta": beta.tolist()},
    }
    for name, item in out.items():
        item["val_spearman"] = float(spearmanr(item["val"], y_val).statistic)
        item["test_spearman"] = float(spearmanr(item["test"], y_test_global).statistic) if False else None
    return out


def conformal_by_score(val_score, test_score, val_abs, test_abs, alpha, bins):
    global_q = qhat(val_abs.reshape(-1), alpha)
    global_metrics = point_metrics(test_abs.reshape(-1), global_q)

    edges = np.quantile(val_score, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    rows, weighted_width, covered, total = [], 0.0, 0, 0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        val_mask = (val_score > lo) & (val_score <= hi)
        test_mask = (test_score > lo) & (test_score <= hi)
        if not val_mask.any() or not test_mask.any():
            continue
        q = qhat(val_abs[val_mask].reshape(-1), alpha)
        test_points = test_abs[test_mask].reshape(-1)
        m = point_metrics(test_points, q)
        global_bin = point_metrics(test_points, global_q)
        rows.append(
            {
                "bin": i + 1,
                "test_samples": int(test_mask.sum()),
                "score_min": float(test_score[test_mask].min()),
                "score_max": float(test_score[test_mask].max()),
                "coverage": m["coverage"],
                "width": m["width"],
                "global_coverage": global_bin["coverage"],
                "global_width": global_bin["width"],
            }
        )
        weighted_width += m["width"] * len(test_points)
        covered += int((test_points <= q).sum())
        total += len(test_points)

    binned = {
        "coverage": float(covered / total),
        "width": float(weighted_width / total),
        "width_reduction_vs_global": float(1.0 - (weighted_width / total) / global_metrics["width"]),
    }
    return global_metrics, binned, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["etth1_timemixer", "etth2_timemixer", "ettm1_timemixer"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--bins", type=int, default=4)
    args = parser.parse_args()

    for case in args.cases:
        cfg = base_args(case)
        import glob

        ckpts = sorted(glob.glob(cfg.ckpt_glob))
        if not ckpts:
            print({"case": case, "error": "checkpoint_not_found", "glob": cfg.ckpt_glob})
            continue
        exp = Exp_Long_Term_Forecast(cfg)
        exp.model.load_state_dict(torch.load(ckpts[-1], map_location=exp.device))
        parr = PARRPreprocessor(cfg).eval()

        x_val, y_val, abs_val = collect_full(exp, parr, "val")
        x_test, y_test, abs_test = collect_full(exp, parr, "test")
        scores = fit_scores_for_val_test(x_val, y_val, x_test)
        for item in scores.values():
            item["test_spearman"] = float(spearmanr(item["test"], y_test).statistic)
        selected = min(scores, key=lambda name: scores[name]["val_spearman"])
        score_item = scores[selected]
        global_m, binned_m, rows = conformal_by_score(
            score_item["val"], score_item["test"], abs_val, abs_test, args.alpha, args.bins
        )
        print(
            {
                "case": case,
                "checkpoint": ckpts[-1],
                "selected": selected,
                "beta": score_item["beta"],
                "val_spearman": score_item["val_spearman"],
                "test_spearman": score_item["test_spearman"],
                "global": global_m,
                "binned": binned_m,
                "bins": rows,
            }
        )


if __name__ == "__main__":
    main()
