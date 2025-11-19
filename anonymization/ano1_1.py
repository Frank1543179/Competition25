#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 创建匿名化数据 CCi.csv 的示例代码
import sys, os
import argparse
import random
from typing import List

import numpy as np
import pandas as pd

# モジュールの相対参照制限を強制的に回避
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'util'))
from pws_data_format import BiDataFrame, CiDataFrame


# === 修改1: 类别变量扰动 (分布保持式) ===
def mutate_categorical(series: pd.Series, p: float) -> pd.Series:
    """按原始分布替换非空单元格，以概率 p 替换为列中的其他值"""
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


def process_int_column(df: pd.DataFrame, col: str, lo: int, hi: int) -> None:
    """整数列随机扰动"""
    if col not in df.columns:
        return
    s_raw = df[col].astype(str)
    is_blank = s_raw.str.strip().eq("")
    blanks_n = int(is_blank.sum())

    s_num = pd.to_numeric(s_raw.where(~is_blank, np.nan), errors="coerce")
    non_na = s_num.dropna()
    if non_na.empty:
        df[col] = s_raw
        return

    vmin = int(np.floor(non_na.min()))
    vmax = int(np.ceil(non_na.max()))

    delta = np.random.randint(lo, hi + 1, size=len(s_num))
    s_num = s_num.add(delta).clip(lower=vmin, upper=vmax)

    fill_vals = np.random.randint(vmin, vmax + 1, size=len(s_num))
    s_num = s_num.where(~is_blank, fill_vals)

    out = s_num.round(0).astype(int).astype(str)

    if blanks_n > 0:
        idx = np.random.choice(len(out), size=blanks_n, replace=False)
        out.iloc[idx] = ""
    df[col] = out


# === 修改2: 数值变量比例噪声 (新增函数) ===
def process_float_scale(df: pd.DataFrame, col: str, scale: float = 0.02, decimals: int = 2):
    """对连续数值列按比例添加噪声 (保持相关性)，空白保持不变"""
    if col not in df.columns:
        return
    s_raw = df[col].astype(str)
    is_blank = s_raw.str.strip().eq("")
    s_num = pd.to_numeric(s_raw.where(~is_blank, np.nan), errors="coerce")
    non_na = s_num.dropna()
    if non_na.empty:
        return

    noise = np.random.normal(0, scale, size=len(s_num))  # ±scale 比例扰动
    s_num = (s_num * (1 + noise)).clip(lower=non_na.min(), upper=non_na.max())
    df[col] = s_num.where(~is_blank, "").round(decimals).astype(object)


# === 修改3: flag 条件翻转 ===
def flip_flag_with_prob(df: pd.DataFrame, col: str, p: float) -> None:
    """以概率 p 翻转 0/1，且仅在 AGE>40 的样本中翻转"""
    if col not in df.columns or "AGE" not in df.columns:
        return
    s_raw = df[col].astype(str)
    age_num = pd.to_numeric(df["AGE"], errors="coerce")

    mask = s_raw.isin(["0", "1"]) & (np.random.rand(len(s_raw)) < p)
    mask &= (age_num > 40)  # 条件翻转：只对中老年人群翻转

    flipped = s_raw.copy()
    flipped.loc[mask & (s_raw == "0")] = "1"
    flipped.loc[mask & (s_raw == "1")] = "0"
    df[col] = flipped


def process_age_add(df: pd.DataFrame, col: str = "AGE",
                    lo: int = -2, hi: int = 2,
                    min_age: int = 2, max_age: int = 110) -> None:
    """AGE列整数扰动"""
    if col not in df.columns:
        return
    s_raw = df[col].astype(str)
    is_blank = s_raw.str.strip().eq("")
    s_num = pd.to_numeric(s_raw.where(~is_blank, np.nan), errors="coerce")
    non_na = s_num.dropna()
    if non_na.empty:
        return

    delta = np.random.randint(lo, hi + 1, size=len(s_num))
    s_num = s_num.add(delta).clip(lower=min_age, upper=max_age).round(0)

    df[col] = s_num.where(~is_blank, "").astype(object).astype(str)


# === 新增功能1: 对年龄小于10岁的特定列进行内部打乱 ===
def shuffle_columns_for_young_age(df: pd.DataFrame, age_threshold: int = 10, seed: int = None) -> None:
    """对年龄小于10岁的样本，在特定列内部进行随机打乱"""
    if "AGE" not in df.columns:
        return

    age_num = pd.to_numeric(df["AGE"], errors="coerce")
    young_mask = age_num < age_threshold

    if not young_mask.any():
        return

    young_indices = df.index[young_mask]

    # 需要打乱的列
    columns_to_shuffle = [
        "AGE", "encounter_count", "num_procedures", "num_medications",
        "num_immunizations", "num_allergies", "num_devices"
    ]

    for col in columns_to_shuffle:
        if col in df.columns:
            # 获取年轻人群的该列数据
            young_data = df.loc[young_mask, col].copy()
            # 随机打乱顺序（使用传入的seed参数）
            shuffled_data = young_data.sample(frac=1, random_state=seed).values
            # 重新赋值
            df.loc[young_mask, col] = shuffled_data


# === 新增功能2: 根据肥胖标志调整BMI和体重 ===
def adjust_bmi_weight_by_obesity(df: pd.DataFrame) -> None:
    """根据obesity_flag调整BMI和体重"""
    if "obesity_flag" not in df.columns:
        return

    # 确保有需要调整的列
    bmi_col = "mean_bmi"
    weight_col = "mean_weight"

    if bmi_col not in df.columns and weight_col not in df.columns:
        return

    # 处理肥胖标志
    obesity_raw = df["obesity_flag"].astype(str)
    is_obese = obesity_raw == "1"
    is_not_obese = obesity_raw == "0"

    # 调整BMI
    if bmi_col in df.columns:
        bmi_raw = df[bmi_col].astype(str)
        is_blank_bmi = bmi_raw.str.strip().eq("")
        bmi_num = pd.to_numeric(bmi_raw.where(~is_blank_bmi, np.nan), errors="coerce")

        # 对肥胖者增加BMI，对非肥胖者减少BMI
        bmi_adjustment = np.zeros(len(bmi_num))
        bmi_adjustment[is_obese] = np.random.uniform(1.0, 3.0, size=is_obese.sum())  # 增加1-3
        bmi_adjustment[is_not_obese] = np.random.uniform(-2.0, -0.5, size=is_not_obese.sum())  # 减少0.5-2

        bmi_num_adjusted = bmi_num + bmi_adjustment
        # 确保BMI在合理范围内
        bmi_num_adjusted = bmi_num_adjusted.clip(lower=10, upper=50)

        df[bmi_col] = bmi_num_adjusted.where(~is_blank_bmi, "").round(2).astype(object)

    # 调整体重
    if weight_col in df.columns:
        weight_raw = df[weight_col].astype(str)
        is_blank_weight = weight_raw.str.strip().eq("")
        weight_num = pd.to_numeric(weight_raw.where(~is_blank_weight, np.nan), errors="coerce")

        # 对肥胖者增加体重，对非肥胖者减少体重
        weight_adjustment = np.zeros(len(weight_num))
        weight_adjustment[is_obese] = np.random.uniform(2.0, 8.0, size=is_obese.sum())  # 增加2-8kg
        weight_adjustment[is_not_obese] = np.random.uniform(-5.0, -1.0, size=is_not_obese.sum())  # 减少1-5kg

        weight_num_adjusted = weight_num + weight_adjustment
        # 确保体重在合理范围内
        weight_num_adjusted = weight_num_adjusted.clip(lower=10, upper=200)

        df[weight_col] = weight_num_adjusted.where(~is_blank_weight, "").round(2).astype(object)


def main():
    parser = argparse.ArgumentParser(description="匿名化します（BIRTHDATE等の日付処理なし）。")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード（再現用、省略可）")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Biを読み込み
    df = BiDataFrame.read_csv(args.input_csv)

    # ---- 新增: 对年龄小于10岁的特定列进行内部打乱 ----
    shuffle_columns_for_young_age(df, age_threshold=10, seed=args.seed)

    # ---- 类别列 ----
    if "GENDER" in df.columns:
        df["GENDER"] = mutate_categorical(df["GENDER"], p=0.01)
    if "RACE" in df.columns:
        df["RACE"] = mutate_categorical(df["RACE"], p=0.015)
    if "ETHNICITY" in df.columns:
        df["ETHNICITY"] = mutate_categorical(df["ETHNICITY"], p=0.01)

    # ---- AGE ----
    process_age_add(df, col="AGE", lo=-2, hi=2, min_age=2, max_age=110)

    # ---- 整数列 ----
    process_int_column(df, "encounter_count", lo=-10, hi=10)
    process_int_column(df, "num_procedures", lo=-10, hi=10)
    process_int_column(df, "num_medications", lo=-5, hi=5)
    process_int_column(df, "num_immunizations", lo=-1, hi=1)
    process_int_column(df, "num_allergies", lo=-1, hi=1)
    process_int_column(df, "num_devices", lo=-1, hi=1)  # 缩小扰动范围

    # ---- flag 列 ----
    # stroke_flag 不进行改变（注释掉或删除相关行）
    flip_flag_with_prob(df, "asthma_flag", p=0.01)
    # flip_flag_with_prob(df, "stroke_flag",     p=0.02)  # 移除stroke_flag的扰动
    flip_flag_with_prob(df, "obesity_flag", p=0.02)
    flip_flag_with_prob(df, "depression_flag", p=0.02)

    # ---- 新增: 根据肥胖标志调整BMI和体重 ----
    adjust_bmi_weight_by_obesity(df)

    # ---- 连续变量比例扰动 ----
    process_float_scale(df, "mean_systolic_bp", scale=0.01, decimals=2)
    process_float_scale(df, "mean_diastolic_bp", scale=0.01, decimals=2)
    # mean_weight和mean_bmi已经在adjust_bmi_weight_by_obesity中处理过，这里不再重复处理
    # process_float_scale(df, "mean_weight",       scale=0.01, decimals=2)
    # process_float_scale(df, "mean_bmi",          scale=0.01, decimals=2)

    # ---- 输出 ----
    Ci_df = CiDataFrame(df)
    Ci_df.to_csv(args.output_csv)


if __name__ == "__main__":
    main()