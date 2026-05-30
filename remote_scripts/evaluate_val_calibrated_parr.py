import argparse
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.getcwd())

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from layers.PARR import PARRPreprocessor


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def corr(x, y):
    return {
        "spearman": float(spearmanr(x, y).statistic),
        "pearson": float(pearsonr(x, y).statistic),
    }


def qbins(score, residual, bins=4):
    order = np.argsort(score)
    rows = []
    for i, idx in enumerate(np.array_split(order, bins), 1):
        rows.append(
            {
                "bin": i,
                "n": int(len(idx)),
                "score_min": float(score[idx].min()),
                "score_max": float(score[idx].max()),
                "mse_mean": float(residual[idx].mean()),
                "mse_p90": float(np.quantile(residual[idx], 0.9)),
            }
        )
    return rows


def base_args(case):
    common = dict(
        task_name="long_term_forecast",
        is_training=0,
        root_path="./dataset/ETT-small/",
        features="M",
        target="OT",
        freq="h",
        checkpoints="./checkpoints/",
        seq_len=96,
        pred_len=96,
        d_layers=1,
        expand=2,
        d_conv=4,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        do_predict=False,
        num_workers=0,
        itr=1,
        train_epochs=10,
        patience=3,
        des="",
        loss="MSE",
        lradj="type1",
        use_amp=False,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices="0,1,2,3",
        gpu_type="cuda",
        inverse=False,
        seasonal_patterns="Monthly",
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_dtw=False,
        top_k=5,
        num_kernels=6,
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
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
    if case == "etth2_timemixer":
        common.update(
            data="ETTh2",
            data_path="ETTh2.csv",
            label_len=0,
            e_layers=2,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=16,
            d_ff=32,
            n_heads=8,
            learning_rate=0.001,
            batch_size=256,
            down_sampling_layers=2,
            down_sampling_method="avg",
            down_sampling_window=2,
            model="TimeMixer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_timemixer_etth2_96_TimeMixer_ETTh2_*crossdata_timemixer_base_0/checkpoint.pth",
        )
    elif case == "etth2_patchtst":
        common.update(
            data="ETTh2",
            data_path="ETTh2.csv",
            label_len=48,
            e_layers=2,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=256,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=256,
            patch_len=16,
            model="PatchTST",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_patchtst_etth2_96_PatchTST_ETTh2_*crossdata_patchtst_base_0/checkpoint.pth",
        )
    elif case == "etth1_timemixer":
        common.update(
            data="ETTh1",
            data_path="ETTh1.csv",
            label_len=0,
            e_layers=2,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=16,
            d_ff=32,
            n_heads=8,
            learning_rate=0.001,
            batch_size=256,
            down_sampling_layers=2,
            down_sampling_method="avg",
            down_sampling_window=2,
            model="TimeMixer",
            ckpt_glob="checkpoints/long_term_forecast_first_real_timemixer_etth1_96_TimeMixer_ETTh1_*first_real_base_0/checkpoint.pth",
        )
    elif case == "etth1_patchtst":
        common.update(
            data="ETTh1",
            data_path="ETTh1.csv",
            label_len=48,
            e_layers=2,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=256,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=256,
            patch_len=16,
            model="PatchTST",
            ckpt_glob="checkpoints/long_term_forecast_patchtst_base_etth1_96_PatchTST_ETTh1_*patchtst_base_0/checkpoint.pth",
        )
    elif case == "etth1_timexer":
        common.update(
            data="ETTh1",
            data_path="ETTh1.csv",
            label_len=48,
            e_layers=1,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=256,
            n_heads=4,
            factor=3,
            learning_rate=0.0005,
            batch_size=256,
            patch_len=16,
            use_norm=1,
            model="TimeXer",
            ckpt_glob="checkpoints/long_term_forecast_timexer_base_etth1_96_TimeXer_ETTh1_*timexer_base_0/checkpoint.pth",
        )
    elif case == "etth1_itransformer":
        common.update(
            data="ETTh1",
            data_path="ETTh1.csv",
            label_len=48,
            e_layers=2,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=256,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=256,
            model="iTransformer",
            ckpt_glob="checkpoints/long_term_forecast_itransformer_base_etth1_96_iTransformer_ETTh1_*itransformer_base_0/checkpoint.pth",
        )
    elif case == "etth2_timexer":
        common.update(
            data="ETTh2",
            data_path="ETTh2.csv",
            label_len=48,
            e_layers=1,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=256,
            n_heads=4,
            factor=3,
            learning_rate=0.0005,
            batch_size=256,
            patch_len=16,
            use_norm=1,
            model="TimeXer",
            ckpt_glob="checkpoints/long_term_forecast_compare_timexer_etth2_96_TimeXer_ETTh2_*compare_timexer_base_0/checkpoint.pth",
        )
    elif case == "etth2_itransformer":
        common.update(
            data="ETTh2",
            data_path="ETTh2.csv",
            label_len=48,
            e_layers=2,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=256,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=256,
            model="iTransformer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_itransformer_etth2_96_iTransformer_ETTh2_*crossdata_itransformer_base_0/checkpoint.pth",
        )
    elif case == "ettm1_timemixer":
        common.update(
            data="ETTm1",
            data_path="ETTm1.csv",
            label_len=0,
            e_layers=2,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=16,
            d_ff=32,
            n_heads=8,
            learning_rate=0.001,
            batch_size=512,
            down_sampling_layers=2,
            down_sampling_method="avg",
            down_sampling_window=2,
            model="TimeMixer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_timemixer_ettm1_96_TimeMixer_ETTm1_*crossdata_timemixer_base_0/checkpoint.pth",
        )
    elif case == "ettm1_patchtst":
        common.update(
            data="ETTm1",
            data_path="ETTm1.csv",
            label_len=48,
            e_layers=2,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=256,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=512,
            patch_len=16,
            model="PatchTST",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_patchtst_ettm1_96_PatchTST_ETTm1_*crossdata_patchtst_base_0/checkpoint.pth",
        )
    elif case == "ettm1_timexer":
        common.update(
            data="ETTm1",
            data_path="ETTm1.csv",
            label_len=48,
            e_layers=1,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=256,
            n_heads=4,
            factor=3,
            learning_rate=0.0005,
            batch_size=512,
            patch_len=16,
            use_norm=1,
            model="TimeXer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_timexer_ettm1_96_TimeXer_ETTm1_*crossdata_timexer_base_0/checkpoint.pth",
        )
    elif case == "ettm1_itransformer":
        common.update(
            data="ETTm1",
            data_path="ETTm1.csv",
            label_len=48,
            e_layers=2,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=256,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=512,
            model="iTransformer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_itransformer_ettm1_96_iTransformer_ETTm1_*crossdata_itransformer_base_0/checkpoint.pth",
        )
    elif case == "ettm2_timemixer":
        common.update(
            data="ETTm2",
            data_path="ETTm2.csv",
            label_len=0,
            e_layers=2,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=16,
            d_ff=32,
            n_heads=8,
            learning_rate=0.001,
            batch_size=512,
            down_sampling_layers=2,
            down_sampling_method="avg",
            down_sampling_window=2,
            model="TimeMixer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_timemixer_ettm2_96_TimeMixer_ETTm2_*crossdata_timemixer_base_0/checkpoint.pth",
        )
    elif case == "ettm2_patchtst":
        common.update(
            data="ETTm2",
            data_path="ETTm2.csv",
            label_len=48,
            e_layers=2,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=256,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=512,
            patch_len=16,
            model="PatchTST",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_patchtst_ettm2_96_PatchTST_ETTm2_*crossdata_patchtst_base_0/checkpoint.pth",
        )
    elif case == "ettm2_timexer":
        common.update(
            data="ETTm2",
            data_path="ETTm2.csv",
            label_len=48,
            e_layers=1,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=256,
            n_heads=4,
            factor=3,
            learning_rate=0.0005,
            batch_size=512,
            patch_len=16,
            use_norm=1,
            model="TimeXer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_timexer_ettm2_96_TimeXer_ETTm2_*crossdata_timexer_base_0/checkpoint.pth",
        )
    elif case == "ettm2_itransformer":
        common.update(
            data="ETTm2",
            data_path="ETTm2.csv",
            label_len=48,
            e_layers=2,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=256,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=512,
            model="iTransformer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_itransformer_ettm2_96_iTransformer_ETTm2_*crossdata_itransformer_base_0/checkpoint.pth",
        )
    elif case == "weather_timemixer":
        common.update(
            root_path="./dataset/weather/",
            data="custom",
            data_path="weather.csv",
            label_len=0,
            e_layers=3,
            enc_in=21,
            dec_in=21,
            c_out=21,
            d_model=16,
            d_ff=32,
            n_heads=8,
            factor=3,
            learning_rate=0.001,
            batch_size=512,
            down_sampling_layers=3,
            down_sampling_method="avg",
            down_sampling_window=2,
            model="TimeMixer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_timemixer_weather_96_TimeMixer_custom_*crossdata_timemixer_weather_0/checkpoint.pth",
        )
    elif case == "weather_patchtst":
        common.update(
            root_path="./dataset/weather/",
            data="custom",
            data_path="weather.csv",
            label_len=48,
            e_layers=2,
            enc_in=21,
            dec_in=21,
            c_out=21,
            d_model=128,
            d_ff=256,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=512,
            patch_len=16,
            model="PatchTST",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_patchtst_weather_96_PatchTST_custom_*crossdata_patchtst_weather_0/checkpoint.pth",
        )
    elif case == "weather_timexer":
        common.update(
            root_path="./dataset/weather/",
            data="custom",
            data_path="weather.csv",
            label_len=48,
            e_layers=1,
            enc_in=21,
            dec_in=21,
            c_out=21,
            d_model=128,
            d_ff=256,
            n_heads=4,
            factor=3,
            learning_rate=0.0005,
            batch_size=256,
            patch_len=16,
            use_norm=1,
            model="TimeXer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_timexer_weather_96_TimeXer_custom_*crossdata_timexer_weather_0/checkpoint.pth",
        )
    elif case == "weather_itransformer":
        common.update(
            root_path="./dataset/weather/",
            data="custom",
            data_path="weather.csv",
            label_len=48,
            e_layers=2,
            enc_in=21,
            dec_in=21,
            c_out=21,
            d_model=128,
            d_ff=256,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=512,
            model="iTransformer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_itransformer_weather_96_iTransformer_custom_*crossdata_itransformer_weather_0/checkpoint.pth",
        )
    elif case == "exchange_timemixer":
        common.update(
            root_path="./dataset/exchange_rate/",
            data="custom",
            data_path="exchange_rate.csv",
            label_len=0,
            e_layers=2,
            enc_in=8,
            dec_in=8,
            c_out=8,
            d_model=16,
            d_ff=32,
            n_heads=8,
            factor=3,
            learning_rate=0.001,
            batch_size=256,
            down_sampling_layers=2,
            down_sampling_method="avg",
            down_sampling_window=2,
            model="TimeMixer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_timemixer_exchange_96_TimeMixer_custom_*crossdata_timemixer_exchange_0/checkpoint.pth",
        )
    elif case == "exchange_patchtst":
        common.update(
            root_path="./dataset/exchange_rate/",
            data="custom",
            data_path="exchange_rate.csv",
            label_len=48,
            e_layers=2,
            enc_in=8,
            dec_in=8,
            c_out=8,
            d_model=128,
            d_ff=256,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=256,
            patch_len=16,
            model="PatchTST",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_patchtst_exchange_96_PatchTST_custom_*crossdata_patchtst_exchange_0/checkpoint.pth",
        )
    elif case == "exchange_timexer":
        common.update(
            root_path="./dataset/exchange_rate/",
            data="custom",
            data_path="exchange_rate.csv",
            label_len=48,
            e_layers=1,
            enc_in=8,
            dec_in=8,
            c_out=8,
            d_model=128,
            d_ff=256,
            n_heads=4,
            factor=3,
            learning_rate=0.0005,
            batch_size=256,
            patch_len=16,
            use_norm=1,
            model="TimeXer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_timexer_exchange_96_TimeXer_custom_*crossdata_timexer_exchange_0/checkpoint.pth",
        )
    elif case == "exchange_itransformer":
        common.update(
            root_path="./dataset/exchange_rate/",
            data="custom",
            data_path="exchange_rate.csv",
            label_len=48,
            e_layers=2,
            enc_in=8,
            dec_in=8,
            c_out=8,
            d_model=128,
            d_ff=256,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=256,
            model="iTransformer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_itransformer_exchange_96_iTransformer_custom_*crossdata_itransformer_exchange_0/checkpoint.pth",
        )
    elif case == "electricity_timemixer":
        common.update(
            root_path="./dataset/electricity/",
            data="custom",
            data_path="electricity.csv",
            label_len=0,
            e_layers=2,
            enc_in=321,
            dec_in=321,
            c_out=321,
            d_model=16,
            d_ff=32,
            n_heads=8,
            factor=3,
            learning_rate=0.001,
            batch_size=32,
            down_sampling_layers=2,
            down_sampling_method="avg",
            down_sampling_window=2,
            model="TimeMixer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_timemixer_electricity_96_TimeMixer_custom_*crossdata_timemixer_electricity_0/checkpoint.pth",
        )
    elif case == "electricity_patchtst":
        common.update(
            root_path="./dataset/electricity/",
            data="custom",
            data_path="electricity.csv",
            label_len=48,
            e_layers=2,
            enc_in=321,
            dec_in=321,
            c_out=321,
            d_model=64,
            d_ff=128,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=16,
            patch_len=16,
            model="PatchTST",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_patchtst_electricity_96_PatchTST_custom_*crossdata_patchtst_electricity_0/checkpoint.pth",
        )
    elif case == "electricity_timexer":
        common.update(
            root_path="./dataset/electricity/",
            data="custom",
            data_path="electricity.csv",
            label_len=48,
            e_layers=1,
            enc_in=321,
            dec_in=321,
            c_out=321,
            d_model=64,
            d_ff=128,
            n_heads=4,
            factor=3,
            learning_rate=0.0005,
            batch_size=16,
            patch_len=16,
            use_norm=1,
            model="TimeXer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_timexer_electricity_96_TimeXer_custom_*crossdata_timexer_electricity_0/checkpoint.pth",
        )
    elif case == "electricity_itransformer":
        common.update(
            root_path="./dataset/electricity/",
            data="custom",
            data_path="electricity.csv",
            label_len=48,
            e_layers=2,
            enc_in=321,
            dec_in=321,
            c_out=321,
            d_model=64,
            d_ff=128,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=16,
            model="iTransformer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_itransformer_electricity_96_iTransformer_custom_*crossdata_itransformer_electricity_0/checkpoint.pth",
        )
    elif case == "traffic_timemixer":
        common.update(
            root_path="./dataset/traffic/",
            data="custom",
            data_path="traffic.csv",
            label_len=0,
            e_layers=3,
            enc_in=862,
            dec_in=862,
            c_out=862,
            d_model=32,
            d_ff=64,
            n_heads=8,
            factor=3,
            learning_rate=0.001,
            batch_size=8,
            down_sampling_layers=3,
            down_sampling_method="avg",
            down_sampling_window=2,
            model="TimeMixer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_timemixer_traffic_96_TimeMixer_custom_*crossdata_timemixer_traffic_0/checkpoint.pth",
        )
    elif case == "traffic_patchtst":
        common.update(
            root_path="./dataset/traffic/",
            data="custom",
            data_path="traffic.csv",
            label_len=48,
            e_layers=2,
            enc_in=862,
            dec_in=862,
            c_out=862,
            d_model=64,
            d_ff=128,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=8,
            patch_len=16,
            model="PatchTST",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_patchtst_traffic_96_PatchTST_custom_*crossdata_patchtst_traffic_0/checkpoint.pth",
        )
    elif case == "traffic_timexer":
        common.update(
            root_path="./dataset/traffic/",
            data="custom",
            data_path="traffic.csv",
            label_len=48,
            e_layers=1,
            enc_in=862,
            dec_in=862,
            c_out=862,
            d_model=64,
            d_ff=128,
            n_heads=4,
            factor=3,
            learning_rate=0.0005,
            batch_size=8,
            patch_len=16,
            use_norm=1,
            model="TimeXer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_timexer_traffic_96_TimeXer_custom_*crossdata_timexer_traffic_0/checkpoint.pth",
        )
    elif case == "traffic_itransformer":
        common.update(
            root_path="./dataset/traffic/",
            data="custom",
            data_path="traffic.csv",
            label_len=48,
            e_layers=2,
            enc_in=862,
            dec_in=862,
            c_out=862,
            d_model=64,
            d_ff=128,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=8,
            model="iTransformer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_itransformer_traffic_96_iTransformer_custom_*crossdata_itransformer_traffic_0/checkpoint.pth",
        )
    elif case == "illness_timemixer":
        common.update(
            root_path="./dataset/illness/",
            data="custom",
            data_path="national_illness.csv",
            freq="w",
            seq_len=36,
            pred_len=24,
            label_len=0,
            e_layers=2,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=16,
            d_ff=32,
            n_heads=8,
            factor=3,
            learning_rate=0.001,
            batch_size=32,
            down_sampling_layers=2,
            down_sampling_method="avg",
            down_sampling_window=2,
            model="TimeMixer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_timemixer_illness_24_TimeMixer_custom_*crossdata_timemixer_illness_0/checkpoint.pth",
        )
    elif case == "illness_patchtst":
        common.update(
            root_path="./dataset/illness/",
            data="custom",
            data_path="national_illness.csv",
            freq="w",
            seq_len=36,
            pred_len=24,
            label_len=18,
            e_layers=2,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=64,
            d_ff=128,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=16,
            patch_len=16,
            model="PatchTST",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_patchtst_illness_24_PatchTST_custom_*crossdata_patchtst_illness_0/checkpoint.pth",
        )
    elif case == "illness_timexer":
        common.update(
            root_path="./dataset/illness/",
            data="custom",
            data_path="national_illness.csv",
            freq="w",
            seq_len=36,
            pred_len=24,
            label_len=18,
            e_layers=1,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=64,
            d_ff=128,
            n_heads=4,
            factor=3,
            learning_rate=0.0005,
            batch_size=16,
            patch_len=16,
            use_norm=1,
            model="TimeXer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_timexer_illness_24_TimeXer_custom_*crossdata_timexer_illness_0/checkpoint.pth",
        )
    elif case == "illness_itransformer":
        common.update(
            root_path="./dataset/illness/",
            data="custom",
            data_path="national_illness.csv",
            freq="w",
            seq_len=36,
            pred_len=24,
            label_len=18,
            e_layers=2,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=64,
            d_ff=128,
            n_heads=4,
            learning_rate=0.0005,
            batch_size=16,
            model="iTransformer",
            ckpt_glob="checkpoints/long_term_forecast_crossdata_itransformer_illness_24_iTransformer_custom_*crossdata_itransformer_illness_0/checkpoint.pth",
        )
    else:
        raise ValueError(case)
    return SimpleNamespace(**common)


def collect(exp, parr, flag):
    _, loader = exp._get_data(flag=flag)
    preds, trues, comps = [], [], []
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
            preds.append(outputs.detach().cpu().numpy())
            trues.append(truth.detach().cpu().numpy())
            comps.append(comp)
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    residual = ((pred - true) ** 2).mean(axis=(1, 2))
    return np.concatenate(comps, axis=0), residual


def fit_scores(x_val, y_val, x_test):
    mu = x_val.mean(axis=0)
    sigma = x_val.std(axis=0) + 1e-8
    z_val = (x_val - mu) / sigma
    z_test = (x_test - mu) / sigma

    signs = []
    for i in range(z_val.shape[1]):
        s = spearmanr(z_val[:, i], y_val).statistic
        signs.append(0.0 if np.isnan(s) or abs(s) < 0.03 else float(np.sign(s)))
    signs = np.asarray(signs)
    sign_score = sigmoid(-(z_test @ signs))

    x1_val = np.c_[np.ones(len(z_val)), z_val]
    x1_test = np.c_[np.ones(len(z_test)), z_test]
    ridge = 1e-3 * np.eye(x1_val.shape[1])
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x1_val.T @ x1_val + ridge, x1_val.T @ y_val)
    ols_score = -(x1_test @ beta)

    return {
        "sign_beta": signs,
        "sign_score": sign_score,
        "ridge_beta": beta,
        "ridge_score": ols_score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["etth2_timemixer", "etth2_patchtst", "etth1_timemixer", "etth1_patchtst"])
    args = parser.parse_args()

    for case in args.cases:
        cfg = base_args(case)
        ckpts = sorted(glob.glob(cfg.ckpt_glob))
        if not ckpts:
            print({"case": case, "error": "checkpoint_not_found", "glob": cfg.ckpt_glob})
            continue
        ckpt = ckpts[-1]
        exp = Exp_Long_Term_Forecast(cfg)
        exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
        parr = PARRPreprocessor(cfg).eval()
        x_val, y_val = collect(exp, parr, "val")
        x_test, y_test = collect(exp, parr, "test")
        scores = fit_scores(x_val, y_val, x_test)
        print({"case": case, "checkpoint": ckpt, "val_n": len(y_val), "test_n": len(y_test), "test_mse": float(y_test.mean())})
        for name in ["sign_score", "ridge_score"]:
            score = scores[name]
            print(
                {
                    "case": case,
                    "score": name,
                    "corr": corr(score, y_test),
                    "beta": scores["sign_beta"].tolist() if name == "sign_score" else scores["ridge_beta"].tolist(),
                    "bins": qbins(score, y_test),
                }
            )


if __name__ == "__main__":
    main()
