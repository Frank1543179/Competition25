#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版匿名化（改进）
目标：在保证有用性（utility）前提下尽量最小化统计差异
对比点：减少大幅度替换、修复mask对齐问题、采用分层填充、降低扰动幅度
"""
import sys
import os
import argparse
import random
from typing import List, Dict

import numpy as np
import pandas as pd

# 将项目 util 路径加入 sys.path（与你原来一致）
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
    """
    对 series 的缺失值或超出范围值进行分层中位数填充（按 group_cols）。
    如果分层为空，退化为整体中位数或 min_val。
    """
    numeric = pd.to_numeric(series, errors='coerce')
    mask_bad = numeric.isna() | (numeric < min_val) | (numeric > max_val)
    if not mask_bad.any():
        return series

    # 计算分组中位数
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
    """
    对整数计数列做小幅扰动（按 fraction），choices 里包含可能的扰动（例如 [-1,0,1]）
    使用 rng 保证可复现
    """
    numeric = pd.to_numeric(series, errors='coerce')
    valid_mask = numeric.notna()
    # 限制最大扰动比例，避免一次性改变太多
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
# 1. 年龄：轻微扰动（修复原代码可能的逻辑问题）
# -------------------------
def slight_age_perturbation(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """对年龄进行轻微扰动，保持年龄分布，范围限定在2~110"""
    if "AGE" not in df.columns:
        return

    age_num = pd.to_numeric(df["AGE"], errors="coerce")
    empty_age_mask = age_num.isna()

    if empty_age_mask.any():
        median_age = int(round(age_num.median())) if not pd.isna(age_num.median()) else 45
        df.loc[empty_age_mask, "AGE"] = str(int(median_age))

    age_num = pd.to_numeric(df["AGE"], errors="coerce").clip(2, 110).astype(int)

    # 对 2% 的样本做 ±1 的扰动（比原脚本更保守）
    perturb_mask = rng.random(len(df)) < 0.02
    indices = df.index[perturb_mask]
    if len(indices) > 0:
        noise = rng.choice([-1, 0, 1], size=len(indices), p=[0.45, 0.1, 0.45])  # 允许小概率不变
        new_vals = np.clip(age_num.loc[indices] + noise, 2, 110).astype(int)
        df.loc[indices, "AGE"] = new_vals.astype(str)

    # 统一格式
    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce").clip(2, 110).round().astype(int).astype(str)

# -------------------------
# 2. 种族合并：基于阈值的弱合并，避免把大量记录一次性合并
# -------------------------
def merge_race_to_other(df: pd.DataFrame, rng: np.random.Generator, small_thresh_ratio: float = 0.005) -> None:
    """将极少类别合并为 other；只在该类别样本数小于阈值时合并；对 hawaiian/native 采用概率合并以减少突变"""
    if "RACE" not in df.columns:
        return

    counts = df["RACE"].value_counts(dropna=True)
    n_total = len(df)
    small_thresh = max(1, int(n_total * small_thresh_ratio))

    # 小类别全部合并
    small_cats = counts[counts <= small_thresh].index.tolist()

    # 但是对 hawaiian/native 采用概率合并（避免把中等数量一刀切）
    probs_map = {}
    for cat in small_cats:
        probs_map[cat] = 1.0  # 小类别直接合并

    for cat in ['hawaiian', 'native']:
        if cat in counts.index and counts[cat] > small_thresh:
            # 如果数量超过阈值，按 70% 的概率合并其成员为 other
            probs_map[cat] = 0.7

    # 执行概率合并
    mask_list = []
    for cat, p in probs_map.items():
        cat_mask = (df["RACE"] == cat)
        if p >= 1.0:
            df.loc[cat_mask, "RACE"] = "other"
        else:
            selected = cat_mask & (rng.random(len(df)) < p)
            df.loc[selected, "RACE"] = "other"

    print(f"种族合并完成（基于阈值{small_thresh}）：部分小类别合并为 'other'")

# -------------------------
# 3. 民族轻微扰动（更保守）
# -------------------------
def slight_ethnicity_perturbation(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """对民族进行轻微扰动，概率更小以降低统计偏差"""
    if "ETHNICITY" not in df.columns:
        return

    # 更保守：nonhispanic -> hispanic 0.5%， hispanic -> nonhispanic 0.3%
    nonhispanic_mask = (df["ETHNICITY"] == "nonhispanic") & (rng.random(len(df)) < 0.005)
    hispanic_mask = (df["ETHNICITY"] == "hispanic") & (rng.random(len(df)) < 0.003)

    df.loc[nonhispanic_mask, "ETHNICITY"] = "hispanic"
    df.loc[hispanic_mask, "ETHNICITY"] = "nonhispanic"

# -------------------------
# 4. 医疗计数指标的保守扰动（修复逻辑并采用分层填充）
# -------------------------
def conservative_medical_counts_perturbation(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """对医疗计数指标进行保守扰动，并对缺失/异常值进行分层填充（按 AGE 分组与 GENDER）"""
    medical_ranges = {
        "encounter_count": (0, 1211),
        "num_procedures": (0, 500),
        "num_medications": (0, 200),
        "num_immunizations": (0, 50),
        "num_allergies": (0, 20),
        "num_devices": (0, 10)
    }

    for col, (min_val, max_val) in medical_ranges.items():
        if col not in df.columns:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")

        # 仅对 1% 的记录做小幅扰动（更保守）
        df[col] = apply_small_integer_perturbation(df[col], fraction=0.01, choices=[-1, 0, 1], min_val=min_val, max_val=max_val, rng=rng)

        # 对缺失/超范围进行分层中位数填充（按年龄段+性别）
        group_cols = []
        if "AGE" in df.columns:
            df['_age_bin'] = _age_bins(df['AGE'])
            group_cols.append('_age_bin')
        if "GENDER" in df.columns:
            group_cols.append('GENDER')
        if len(group_cols) == 0:
            # 无分层信息时退化为整体中位数
            median_val = int(np.clip(round(numeric.median() if not pd.isna(numeric.median()) else min_val), min_val, max_val))
            mask_bad = numeric.isna() | (numeric < min_val) | (numeric > max_val)
            df.loc[mask_bad, col] = str(median_val)
        else:
            df[col] = stratified_impute(df[col], df, min_val=min_val, max_val=max_val, group_cols=group_cols)
        if '_age_bin' in df.columns:
            df.drop(columns=['_age_bin'], inplace=True)

# -------------------------
# 5. 疾病标志位的保守扰动（不强制目标患病率）
# -------------------------
def conservative_flag_perturbation(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """对疾病标志位进行很保守的扰动：小概率翻转且优先在可合理的年龄段内修改"""
    # asthma_flag: 0.5% 的小概率翻转
    if "asthma_flag" in df.columns:
        mask_valid = df["asthma_flag"].isin(["0", "1"])
        flip_mask = mask_valid & (rng.random(len(df)) < 0.005)
        df.loc[flip_mask, "asthma_flag"] = df.loc[flip_mask, "asthma_flag"].map({"0": "1", "1": "0"})

    # depression_flag: 仅微量增加/减少（总改动不超过 0.5%）
    if "depression_flag" in df.columns:
        current_rate = (df["depression_flag"] == "1").mean()
        max_change = max(1, int(len(df) * 0.005))
        # 随机选择 up to max_change 个样本改变状态（优先年龄 18-40）
        age_num = pd.to_numeric(df["AGE"], errors="coerce")
        candidates = df[(df["depression_flag"] == "0") & (age_num.between(18, 40, inclusive='both'))].index.tolist()
        n_to_add = min(len(candidates), max_change)
        if n_to_add > 0:
            sel = rng.choice(candidates, size=n_to_add, replace=False)
            df.loc[sel, "depression_flag"] = "1"
            print(f"增加了 {len(sel)} 个抑郁症病例（保守策略）")

    # stroke_flag & obesity_flag: 基于年龄段的微量增加
    if "AGE" in df.columns:
        age_num = pd.to_numeric(df["AGE"], errors="coerce")
        if "stroke_flag" in df.columns:
            old_no_stroke_mask = (age_num > 75) & (df["stroke_flag"] == "0") & (rng.random(len(df)) < 0.002)
            df.loc[old_no_stroke_mask, "stroke_flag"] = "1"

        if "obesity_flag" in df.columns:
            middle_no_obese_mask = (age_num >= 40) & (age_num <= 60) & (df["obesity_flag"] == "0") & (rng.random(len(df)) < 0.002)
            df.loc[middle_no_obese_mask, "obesity_flag"] = "1"

# -------------------------
# 6. 生理指标的保守扰动（修复掩码对齐）
# -------------------------
def conservative_physiological_perturbation(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """对生理指标进行保守扰动，修复索引对齐问题，使用分层填充缺失"""
    physiological_ranges = {
        "mean_systolic_bp": (60, 200),
        "mean_diastolic_bp": (40, 120),
        "mean_weight": (30, 200),
        "mean_bmi": (15, 50)
    }

    # systolic bp：对 1.5% 的记录做 ±1 mmHg 的扰动
    if "mean_systolic_bp" in df.columns:
        s = pd.to_numeric(df["mean_systolic_bp"], errors="coerce")
        valid_mask = s.notna()
        perturb_mask = valid_mask & (rng.random(len(df)) < 0.015)
        idx = s.index[perturb_mask]
        if len(idx) > 0:
            noise = rng.choice([-1, 0, 1], size=len(idx), p=[0.45, 0.1, 0.45])
            s.loc[idx] = (s.loc[idx] + noise).clip(*physiological_ranges["mean_systolic_bp"])
            df.loc[idx, "mean_systolic_bp"] = s.loc[idx].round(0).astype(int).astype(str)

        # 缺失/异常分层填充
        group_cols = []
        if "AGE" in df.columns:
            df['_age_bin'] = _age_bins(df['AGE'])
            group_cols.append('_age_bin')
        if "GENDER" in df.columns:
            group_cols.append('GENDER')
        if len(group_cols) > 0:
            df["mean_systolic_bp"] = stratified_impute(df["mean_systolic_bp"], df, *physiological_ranges["mean_systolic_bp"], group_cols=group_cols)
        else:
            # 整体中位数替换
            median_val = int(np.clip(round(s.median() if not pd.isna(s.median()) else physiological_ranges["mean_systolic_bp"][0]), *physiological_ranges["mean_systolic_bp"]))
            mask_bad = s.isna() | (s < physiological_ranges["mean_systolic_bp"][0]) | (s > physiological_ranges["mean_systolic_bp"][1])
            df.loc[mask_bad, "mean_systolic_bp"] = str(median_val)
        if '_age_bin' in df.columns:
            df.drop(columns=['_age_bin'], inplace=True)

    # mean_weight：对 1% 的记录做 ±0.3 kg 的扰动
    if "mean_weight" in df.columns:
        w = pd.to_numeric(df["mean_weight"], errors="coerce")
        valid_mask = w.notna()
        perturb_mask = valid_mask & (rng.random(len(df)) < 0.01)
        idx = w.index[perturb_mask]
        if len(idx) > 0:
            noise = rng.choice([-0.3, 0.0, 0.3], size=len(idx), p=[0.45, 0.1, 0.45])
            w.loc[idx] = (w.loc[idx] + noise).clip(*physiological_ranges["mean_weight"])
            df.loc[idx, "mean_weight"] = w.loc[idx].round(1).astype(str)
        # 填充缺失（分层）
        if "AGE" in df.columns or "GENDER" in df.columns:
            group_cols = []
            if "AGE" in df.columns:
                df['_age_bin_w'] = _age_bins(df['AGE'])
                group_cols.append('_age_bin_w')
            if "GENDER" in df.columns:
                group_cols.append('GENDER')
            df["mean_weight"] = stratified_impute(df["mean_weight"], df, *physiological_ranges["mean_weight"], group_cols=group_cols)
            if '_age_bin_w' in df.columns:
                df.drop(columns=['_age_bin_w'], inplace=True)

    # mean_bmi：对 1% 的记录做 ±0.1 的扰动
    if "mean_bmi" in df.columns:
        b = pd.to_numeric(df["mean_bmi"], errors="coerce")
        valid_mask = b.notna()
        perturb_mask = valid_mask & (rng.random(len(df)) < 0.01)
        idx = b.index[perturb_mask]
        if len(idx) > 0:
            noise = rng.choice([-0.1, 0.0, 0.1], size=len(idx), p=[0.45, 0.1, 0.45])
            b.loc[idx] = (b.loc[idx] + noise).clip(*physiological_ranges["mean_bmi"])
            df.loc[idx, "mean_bmi"] = b.loc[idx].round(1).astype(str)
        # 分层填充
        if "AGE" in df.columns or "GENDER" in df.columns:
            group_cols = []
            if "AGE" in df.columns:
                df['_age_bin_b'] = _age_bins(df['AGE'])
                group_cols.append('_age_bin_b')
            if "GENDER" in df.columns:
                group_cols.append('GENDER')
            df["mean_bmi"] = stratified_impute(df["mean_bmi"], df, *physiological_ranges["mean_bmi"], group_cols=group_cols)
            if '_age_bin_b' in df.columns:
                df.drop(columns=['_age_bin_b'], inplace=True)

# -------------------------
# 7. 性别比例微调（更保守：最多改变总体的0.5%）
# -------------------------
def adjust_gender_ratio(df: pd.DataFrame, rng: np.random.Generator, target_male_ratio: float = 0.5116) -> None:
    """微调性别比例，但最大改变量受限（默认 0.5%）"""
    if "GENDER" not in df.columns:
        return

    current_male_ratio = (df["GENDER"] == "M").mean()
    n_total = len(df)
    n_target_male = int(n_total * target_male_ratio)
    n_current_male = (df["GENDER"] == "M").sum()
    delta = n_target_male - n_current_male

    max_allowed = max(1, int(n_total * 0.005))  # 最多改变 0.5% 的样本
    if abs(delta) == 0:
        return

    n_change = min(abs(delta), max_allowed)
    if delta < 0:
        # 需要减少男性数量：随机选择一部分男性改为 F（优先从缺失或语义不强的记录）
        male_indices = df[df["GENDER"] == "M"].index.tolist()
        if len(male_indices) > 0:
            change_indices = rng.choice(male_indices, size=min(n_change, len(male_indices)), replace=False)
            df.loc[change_indices, "GENDER"] = "F"
    else:
        female_indices = df[df["GENDER"] == "F"].index.tolist()
        if len(female_indices) > 0:
            change_indices = rng.choice(female_indices, size=min(n_change, len(female_indices)), replace=False)
            df.loc[change_indices, "GENDER"] = "M"

# -------------------------
# 8. 最后数据格式清理（更稳健）
# -------------------------
def clean_data_format(df: pd.DataFrame) -> None:
    """确保数据符合 CiDataFrame 的格式要求，类型转换与填充更稳健"""
    # AGE
    if "AGE" in df.columns:
        age_num = pd.to_numeric(df["AGE"], errors="coerce")
        empty_age_mask = age_num.isna()
        if empty_age_mask.any():
            median_age = int(round(age_num.median())) if not pd.isna(age_num.median()) else 45
            df.loc[empty_age_mask, "AGE"] = str(int(median_age))

        age_num = pd.to_numeric(df["AGE"], errors="coerce").round().astype(int)
        age_num = age_num.clip(lower=2, upper=110)
        df["AGE"] = age_num.astype(str)

    numeric_cols = [
        "encounter_count", "num_procedures", "num_medications",
        "num_immunizations", "num_allergies", "num_devices",
        "mean_systolic_bp", "mean_diastolic_bp", "mean_weight"
    ]
    float_cols = ["mean_bmi"]

    for col in numeric_cols:
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            median_val = numeric_series.median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = numeric_series.apply(lambda x: str(int(round(median_val))) if pd.isna(x) else str(int(round(x))))

    for col in float_cols:
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            median_val = numeric_series.median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = numeric_series.apply(lambda x: f"{median_val:.1f}" if pd.isna(x) else f"{x:.1f}")

    categorical_cols = ["GENDER", "RACE", "ETHNICITY"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": "", "None": "", "": ""})
            if df[col].isna().any() or (df[col] == "").any():
                most_common = df[col].mode()
                if len(most_common) > 0:
                    fill_value = most_common.iloc[0]
                    df[col] = df[col].replace({"": fill_value}).fillna(fill_value)

    flag_cols = ["asthma_flag", "stroke_flag", "obesity_flag", "depression_flag"]
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": "", "None": "", "": ""})
            valid_mask = df[col].isin(["0", "1"])
            df.loc[~valid_mask, col] = "0"

# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="改进版匿名化：最小化统计差异，提高有用性（保守策略）")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（再现用）")
    args = parser.parse_args()

    # 使用 numpy 的 Generator，便于控制与替换
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 读取数据
    df = BiDataFrame.read_csv(args.input_csv)
    print(f"原始数据行数: {len(df)}")

    print("开始应用改进的匿名化策略...")

    slight_age_perturbation(df, rng)
    merge_race_to_other(df, rng, small_thresh_ratio=0.003)  # 阈值可调（默认为 0.3%）
    slight_ethnicity_perturbation(df, rng)
    conservative_medical_counts_perturbation(df, rng)
    conservative_flag_perturbation(df, rng)
    conservative_physiological_perturbation(df, rng)
    adjust_gender_ratio(df, rng, target_male_ratio=0.5116)
    clean_data_format(df)

    # 输出
    Ci_df = CiDataFrame(df)
    Ci_df.to_csv(args.output_csv)
    print(f"✅ 改进版匿名化完成！输出文件: {args.output_csv}")
    print(f"处理后数据行数: {len(df)}")

    # 简单统计输出，便于评估
    if "RACE" in df.columns:
        race_counts = df["RACE"].value_counts()
        print("种族分布统计:")
        for race, count in race_counts.items():
            print(f"  {race}: {count}人 ({count / len(df) * 100:.2f}%)")

    if "depression_flag" in df.columns:
        depression_rate = (df["depression_flag"] == "1").mean()
        print(f"抑郁症患病率: {depression_rate * 100:.2f}%")

if __name__ == "__main__":
    main()
