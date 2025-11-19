#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
import argparse
import random
import numpy as np
import pandas as pd

# 相对路径导入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'util'))
from pws_data_format import BiDataFrame, CiDataFrame

# === 类别变量扰动 ===
def mutate_categorical(series: pd.Series, p: float) -> pd.Series:
    s = series.astype(str)
    values = s[s != ""]
    if values.empty:
        return s
    uniq, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    mask = (s != "") & (np.random.rand(len(s)) < p)

    def pick_other(val: str) -> str:
        choices = uniq[uniq != val]
        weights = probs[uniq != val]
        if len(choices) == 0:
            return val
        return np.random.choice(choices, p=weights / weights.sum())

    s.loc[mask] = s.loc[mask].map(pick_other)
    return s

# === 年龄分段扰动 ===
def process_age_segmented(df: pd.DataFrame, col="AGE"):
    if col not in df.columns:
        return
    s_raw = pd.to_numeric(df[col], errors="coerce")
    out = s_raw.copy()

    # 年轻人 ±3
    mask1 = s_raw <= 30
    out.loc[mask1] = out.loc[mask1] + np.random.randint(-3, 4, mask1.sum())

    # 中年人 ±5
    mask2 = (s_raw > 30) & (s_raw <= 60)
    out.loc[mask2] = out.loc[mask2] + np.random.randint(-5, 6, mask2.sum())

    # 老年人 ±2
    mask3 = s_raw > 60
    out.loc[mask3] = out.loc[mask3] + np.random.randint(-2, 3, mask3.sum())

    out = out.clip(2, 110).round(0).astype("Int64").astype(str)
    df[col] = out

# === 整数列比例扰动 ===
def process_int_scale(df: pd.DataFrame, col: str, scale: float = 0.05):
    if col not in df.columns:
        return
    s_raw = pd.to_numeric(df[col], errors="coerce")
    non_na = s_raw.dropna()
    if non_na.empty:
        return
    noise = np.random.normal(0, scale, size=len(s_raw))
    s_new = (s_raw * (1 + noise)).round(0)
    s_new = s_new.clip(lower=non_na.min(), upper=non_na.max())
    df[col] = s_new.astype("Int64").astype(str)

# === 连续变量比例扰动 ===
def process_float_scale(df: pd.DataFrame, col: str, scale: float, decimals: int = 1):
    if col not in df.columns:
        return
    s_raw = pd.to_numeric(df[col], errors="coerce")
    non_na = s_raw.dropna()
    if non_na.empty:
        return
    noise = np.random.normal(0, scale, size=len(s_raw))
    s_new = (s_raw * (1 + noise)).clip(lower=non_na.min(), upper=non_na.max())
    df[col] = s_new.round(decimals).astype(object)

# === flag 条件翻转 ===
def flip_flag_conditional(df: pd.DataFrame, col: str, p: float, cond=None):
    if col not in df.columns:
        return
    s_raw = df[col].astype(str).copy()
    mask = s_raw.isin(["0", "1"])
    if cond is not None:
        mask &= cond
    flip_mask = mask & (np.random.rand(len(s_raw)) < p)
    s_raw.loc[flip_mask & (s_raw == "0")] = "1"
    s_raw.loc[flip_mask & (s_raw == "1")] = "0"
    df[col] = s_raw

# === 主程序 ===
def main():
    parser = argparse.ArgumentParser(description="改进版匿名化")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    df = BiDataFrame.read_csv(args.input_csv)

    # 类别列
    if "GENDER" in df.columns:
        df["GENDER"] = mutate_categorical(df["GENDER"], p=0.01)
    if "RACE" in df.columns:
        df["RACE"] = mutate_categorical(df["RACE"], p=0.02)
    if "ETHNICITY" in df.columns:
        df["ETHNICITY"] = mutate_categorical(df["ETHNICITY"], p=0.015)

    # AGE
    process_age_segmented(df, "AGE")

    # 整数列比例扰动
    for col in ["encounter_count", "num_procedures", "num_medications"]:
        process_int_scale(df, col, scale=0.05)
    for col in ["num_immunizations", "num_allergies", "num_devices"]:
        process_int_scale(df, col, scale=0.02)

    # 连续变量
    process_float_scale(df, "mean_systolic_bp",  scale=0.005, decimals=1)
    process_float_scale(df, "mean_diastolic_bp", scale=0.005, decimals=1)
    process_float_scale(df, "mean_weight",       scale=0.01,  decimals=1)
    process_float_scale(df, "mean_bmi",          scale=0.015, decimals=1)

    # Flag 条件翻转
    if "AGE" in df.columns:
        age_num = pd.to_numeric(df["AGE"], errors="coerce")
        flip_flag_conditional(df, "asthma_flag",  p=0.01, cond=(age_num < 20))
        flip_flag_conditional(df, "stroke_flag",  p=0.01, cond=(age_num > 60))
    flip_flag_conditional(df, "obesity_flag",    p=0.01)
    flip_flag_conditional(df, "depression_flag", p=0.01)

    # 输出
    Ci_df = CiDataFrame(df)
    Ci_df.to_csv(args.output_csv)

if __name__ == "__main__":
    main()
