#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 创建匿名化数据 CCi.csv (针对 BB07_1.csv)
import sys, os
import argparse
import random
import numpy as np
import pandas as pd

# 相対参照を回避
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'util'))
from pws_data_format import BiDataFrame, CiDataFrame

# === 类别扰动（分布保持）===
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

# === 整数扰动（通用）===
def process_int_column(df: pd.DataFrame, col: str, lo: int, hi: int, min_val: int = 0) -> None:
    if col not in df.columns:
        return
    s_raw = df[col].astype(str)
    is_blank = s_raw.str.strip().eq("")
    blanks_n = int(is_blank.sum())
    s_num = pd.to_numeric(s_raw.where(~is_blank, np.nan), errors="coerce")

    if s_num.dropna().empty:
        df[col] = s_raw
        return

    delta = np.random.randint(lo, hi + 1, size=len(s_num))
    s_num = (s_num + delta).clip(lower=min_val)

    out = s_num.round(0).astype(int).astype(str)
    if blanks_n > 0:
        idx = np.random.choice(len(out), size=blanks_n, replace=False)
        out.iloc[idx] = ""
    df[col] = out

# === num_allergies 特殊处理 ===
def process_allergies(df: pd.DataFrame, col: str = "num_allergies") -> None:
    if col not in df.columns:
        return
    s = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    mask = s >= 1
    noise = np.random.randint(-1, 2, size=mask.sum())  # -1,0,1
    s.loc[mask] = np.maximum(1, s.loc[mask] + noise)
    df[col] = s.astype(str)

# === AGE扰动 ===
def process_age_add(df: pd.DataFrame, col: str = "AGE", lo: int = -1, hi: int = 1,
                    min_age: int = 2, max_age: int = 110) -> None:
    if col not in df.columns:
        return
    s_raw = df[col].astype(str)
    is_blank = s_raw.str.strip().eq("")
    s_num = pd.to_numeric(s_raw.where(~is_blank, np.nan), errors="coerce")
    if s_num.dropna().empty:
        return
    delta = np.random.randint(lo, hi + 1, size=len(s_num))
    s_num = s_num.add(delta).clip(lower=min_age, upper=max_age).round(0)
    df[col] = s_num.where(~is_blank, "").astype(object).astype(str)

# === 连续变量比例噪声 ===
def process_float_scale(df: pd.DataFrame, col: str, scale: float = 0.005, decimals: int = 2):
    if col not in df.columns:
        return
    s_raw = df[col].astype(str)
    is_blank = s_raw.str.strip().eq("")
    s_num = pd.to_numeric(s_raw.where(~is_blank, np.nan), errors="coerce")
    if s_num.dropna().empty:
        return
    noise = np.random.normal(0, scale, size=len(s_num))
    s_num = (s_num * (1 + noise)).clip(lower=s_num.min(), upper=s_num.max())
    df[col] = s_num.where(~is_blank, "").round(decimals).astype(object)

# === flag 条件翻转 ===
def flip_flag_with_prob(df: pd.DataFrame, col: str, p: float) -> None:
    if col not in df.columns or "AGE" not in df.columns:
        return
    s_raw = df[col].astype(str)
    age_num = pd.to_numeric(df["AGE"], errors="coerce")
    mask = s_raw.isin(["0", "1"]) & (np.random.rand(len(s_raw)) < p) & (age_num > 40)
    flipped = s_raw.copy()
    flipped.loc[mask & (s_raw == "0")] = "1"
    flipped.loc[mask & (s_raw == "1")] = "0"
    df[col] = flipped

# === 主程序 ===
def main():
    parser = argparse.ArgumentParser(description="匿名化处理（针对 BB07_1.csv → CCi.csv）")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    df = BiDataFrame.read_csv(args.input_csv)

    # 类别列
    if "GENDER" in df.columns:
        df["GENDER"] = mutate_categorical(df["GENDER"], p=0.003)
    if "RACE" in df.columns:
        df["RACE"] = mutate_categorical(df["RACE"], p=0.005)
    if "ETHNICITY" in df.columns:
        df["ETHNICITY"] = mutate_categorical(df["ETHNICITY"], p=0.005)

    # AGE
    process_age_add(df, col="AGE", lo=-1, hi=1)

    # 整数列
    process_int_column(df, "encounter_count",  lo=-3, hi=3)
    process_int_column(df, "num_procedures",   lo=-3, hi=3)
    process_int_column(df, "num_medications",  lo=-2, hi=2)
    process_int_column(df, "num_immunizations",lo=-1, hi=1)
    process_allergies(df, "num_allergies")
    process_int_column(df, "num_devices",      lo=-1, hi=1)

    # flag列（保留 asthma_flag, stroke_flag）
    flip_flag_with_prob(df, "obesity_flag",    p=0.01)
    flip_flag_with_prob(df, "depression_flag", p=0.01)

    # 连续变量
    process_float_scale(df, "mean_systolic_bp",  scale=0.005, decimals=2)
    process_float_scale(df, "mean_diastolic_bp", scale=0.005, decimals=2)
    process_float_scale(df, "mean_weight",       scale=0.005, decimals=2)
    process_float_scale(df, "mean_bmi",          scale=0.01, decimals=2)

    # 输出
    Ci_df = CiDataFrame(df)
    Ci_df.to_csv(args.output_csv)

if __name__ == "__main__":
    main()
