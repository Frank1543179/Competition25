#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版匿名化（改进）
目标：在保证有用性（utility）前提下尽量最小化统计差异
对比点：减少大幅度替换、修复mask对齐问题、采用分层填充、降低扰动幅度

特别说明：
- mean_systolic_bp, mean_diastolic_bp, mean_weight：保留整数（医学常用记录格式）
- mean_bmi：保留 1 位小数（BMI 常用记录格式）
"""

import sys
import os
import argparse
import random
from typing import List

import numpy as np
import pandas as pd

# 将项目 util 路径加入 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'util'))
from pws_data_format import BiDataFrame, CiDataFrame

# -------------------------
# helper 函数
# -------------------------
def _age_bins(series: pd.Series) -> pd.Series:
    """把年龄分成几个区间，便于分层填充"""
    age = pd.to_numeric(series, errors="coerce")
    bins = [0, 18, 30, 40, 50, 65, 80, 120]
    labels = ['<=18', '19-30', '31-40', '41-50', '51-65', '66-80', '80+']
    return pd.cut(age.fillna(age.median()), bins=bins, labels=labels, include_lowest=True)

def stratified_impute(series: pd.Series, df: pd.DataFrame, min_val: float, max_val: float, group_cols: List[str]) -> pd.Series:
    """分层中位数填充，保证值落在范围内"""
    numeric = pd.to_numeric(series, errors='coerce')
    mask_bad = numeric.isna() | (numeric < min_val) | (numeric > max_val)
    if not mask_bad.any():
        return series

    groups = df[group_cols].copy()
    groups['_val'] = numeric
    group_meds = groups.groupby(group_cols)['_val'].median()

    fill_values = {}
    for idx in groups[mask_bad].index:
        key = tuple(groups.loc[idx, col] for col in group_cols)
        med = group_meds.get(key, np.nan)
        if pd.isna(med):
            med = numeric.median()
        if pd.isna(med):
            med = min_val
        fill_values[idx] = int(np.clip(round(med), min_val, max_val))

    out = series.copy()
    for idx, val in fill_values.items():
        out.at[idx] = str(int(val))
    return out

def apply_small_integer_perturbation(series: pd.Series, fraction: float, choices: List[int], min_val: int, max_val: int, rng: np.random.Generator) -> pd.Series:
    """对整数计数列做小幅扰动"""
    numeric = pd.to_numeric(series, errors='coerce')
    valid_mask = numeric.notna()
    perturb_mask = valid_mask & (rng.random(len(numeric)) < fraction)
    idx = numeric.index[perturb_mask]
    if len(idx) == 0:
        return series

    noise = rng.choice(choices, size=len(idx))
    new_values = (numeric.loc[idx] + noise).clip(min_val, max_val).round().astype(int)
    out = series.copy()
    out.loc[idx] = new_values.astype(str)
    return out

# -------------------------
# 年龄：轻微扰动
# -------------------------
def slight_age_perturbation(df: pd.DataFrame, rng: np.random.Generator) -> None:
    if "AGE" not in df.columns:
        return
    age_num = pd.to_numeric(df["AGE"], errors="coerce")
    empty_age_mask = age_num.isna()
    if empty_age_mask.any():
        median_age = int(round(age_num.median())) if not pd.isna(age_num.median()) else 45
        df.loc[empty_age_mask, "AGE"] = str(int(median_age))

    age_num = pd.to_numeric(df["AGE"], errors="coerce").clip(2, 110).astype(int)
    perturb_mask = rng.random(len(df)) < 0.02
    indices = df.index[perturb_mask]
    if len(indices) > 0:
        noise = rng.choice([-1, 0, 1], size=len(indices), p=[0.45, 0.1, 0.45])
        new_vals = np.clip(age_num.loc[indices] + noise, 2, 110).astype(int)
        df.loc[indices, "AGE"] = new_vals.astype(str)

    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce").clip(2, 110).round().astype(int).astype(str)

# -------------------------
# 生理指标扰动
# -------------------------
def conservative_physiological_perturbation(df: pd.DataFrame, rng: np.random.Generator) -> None:
    physiological_ranges = {
        "mean_systolic_bp": (60, 200),   # 血压 mmHg -> 整数
        "mean_diastolic_bp": (40, 120),  # 血压 mmHg -> 整数
        "mean_weight": (30, 200),        # 体重 kg -> 整数
        "mean_bmi": (15, 50)             # BMI -> 小数，保留1位
    }

    # mean_systolic_bp
    if "mean_systolic_bp" in df.columns:
        s = pd.to_numeric(df["mean_systolic_bp"], errors="coerce")
        perturb_mask = s.notna() & (rng.random(len(df)) < 0.015)
        idx = s.index[perturb_mask]
        if len(idx) > 0:
            noise = rng.choice([-1, 0, 1], size=len(idx))
            s.loc[idx] = (s.loc[idx] + noise).clip(*physiological_ranges["mean_systolic_bp"])
            df.loc[idx, "mean_systolic_bp"] = s.loc[idx].round(0).astype(int).astype(str)

    # mean_diastolic_bp
    if "mean_diastolic_bp" in df.columns:
        d = pd.to_numeric(df["mean_diastolic_bp"], errors="coerce")
        perturb_mask = d.notna() & (rng.random(len(df)) < 0.015)
        idx = d.index[perturb_mask]
        if len(idx) > 0:
            noise = rng.choice([-1, 0, 1], size=len(idx))
            d.loc[idx] = (d.loc[idx] + noise).clip(*physiological_ranges["mean_diastolic_bp"])
            df.loc[idx, "mean_diastolic_bp"] = d.loc[idx].round(0).astype(int).astype(str)

    # mean_weight
    if "mean_weight" in df.columns:
        w = pd.to_numeric(df["mean_weight"], errors="coerce")
        perturb_mask = w.notna() & (rng.random(len(df)) < 0.01)
        idx = w.index[perturb_mask]
        if len(idx) > 0:
            noise = rng.choice([-0.3, 0.0, 0.3], size=len(idx))
            w.loc[idx] = (w.loc[idx] + noise).clip(*physiological_ranges["mean_weight"])
            df.loc[idx, "mean_weight"] = w.loc[idx].round(0).astype(int).astype(str)

    # mean_bmi
    if "mean_bmi" in df.columns:
        b = pd.to_numeric(df["mean_bmi"], errors="coerce")
        perturb_mask = b.notna() & (rng.random(len(df)) < 0.01)
        idx = b.index[perturb_mask]
        if len(idx) > 0:
            noise = rng.choice([-0.1, 0.0, 0.1], size=len(idx))
            b.loc[idx] = (b.loc[idx] + noise).clip(*physiological_ranges["mean_bmi"])
            df.loc[idx, "mean_bmi"] = b.loc[idx].round(1).astype(str)  # BMI 保留 1 位小数

# -------------------------
# 数据格式清理
# -------------------------
def clean_data_format(df: pd.DataFrame) -> None:
    """确保格式：AGE int, 计数列 int, 体重血压 int, BMI 一位小数"""
    if "AGE" in df.columns:
        df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce").clip(2, 110).round().astype(int).astype(str)

    numeric_cols = ["encounter_count", "num_procedures", "num_medications",
                    "num_immunizations", "num_allergies", "num_devices",
                    "mean_systolic_bp", "mean_diastolic_bp", "mean_weight"]
    float_cols = ["mean_bmi"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round().astype(int).astype(str)

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).round(1).astype(str)

# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="改进版匿名化：最小化统计差异，提高有用性")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    df = BiDataFrame.read_csv(args.input_csv)
    print(f"原始数据行数: {len(df)}")

    slight_age_perturbation(df, rng)
    conservative_physiological_perturbation(df, rng)
    clean_data_format(df)

    Ci_df = CiDataFrame(df)
    Ci_df.to_csv(args.output_csv)
    print(f"✅ 改进版匿名化完成！输出文件: {args.output_csv}")

if __name__ == "__main__":
    main()
