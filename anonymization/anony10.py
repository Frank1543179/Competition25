#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 优化版匿名化：平衡匿名性和有用性
import sys, os
import argparse
import random
from typing import List, Dict

import numpy as np
import pandas as pd

# モジュールの相対参照制限を強制的に回避
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'util'))
from pws_data_format import BiDataFrame, CiDataFrame


# === 1. 优化年龄组互换策略 ===
def swap_age_groups(df: pd.DataFrame) -> None:
    """对相似年龄组进行温和互换"""
    if "AGE" not in df.columns:
        return

    # 确保年龄没有空值
    age_num = pd.to_numeric(df["AGE"], errors="coerce")
    empty_age_mask = age_num.isna()

    if empty_age_mask.any():
        median_age = age_num.median()
        if pd.isna(median_age):
            median_age = 45
        df.loc[empty_age_mask, "AGE"] = str(int(median_age))

    age_num = pd.to_numeric(df["AGE"], errors="coerce")

    # 温和的年龄组互换
    age_swap_groups = [
        ((21, 30), (31, 40), 0.06),  # 降低比例
        ((51, 60), (61, 70), 0.06),
    ]

    for (young_min, young_max), (old_min, old_max), swap_ratio in age_swap_groups:
        young_mask = (age_num >= young_min) & (age_num <= young_max)
        old_mask = (age_num >= old_min) & (age_num <= old_max)

        young_indices = df[young_mask].index.tolist()
        old_indices = df[old_mask].index.tolist()

        n_swap = min(int(len(young_indices) * swap_ratio), int(len(old_indices) * swap_ratio))

        if n_swap > 0:
            swap_young = random.sample(young_indices, n_swap)
            swap_old = random.sample(old_indices, n_swap)

            young_ages = df.loc[swap_young, "AGE"].copy()
            old_ages = df.loc[swap_old, "AGE"].copy()

            df.loc[swap_young, "AGE"] = old_ages
            df.loc[swap_old, "AGE"] = young_ages


# === 2. 优化种族合并策略 ===
def merge_race_to_other(df: pd.DataFrame) -> None:
    """只合并极少数种族类别"""
    if "RACE" not in df.columns:
        return

    # 只合并数量最少的种族
    races_to_merge = ['hawaiian', 'native']

    original_counts = df["RACE"].value_counts()

    # 只合并数量很少的种族
    mask = df["RACE"].isin(races_to_merge)
    df.loc[mask, "RACE"] = "other"


# === 3. 优化民族扰动 ===
def safe_ethnicity_perturbation(df: pd.DataFrame) -> None:
    """降低民族扰动的强度"""
    if "ETHNICITY" not in df.columns:
        return

    # 降低扰动比例
    nonhispanic_mask = (df["ETHNICITY"] == "nonhispanic") & (np.random.rand(len(df)) < 0.04)
    hispanic_mask = (df["ETHNICITY"] == "hispanic") & (np.random.rand(len(df)) < 0.03)

    df.loc[nonhispanic_mask, "ETHNICITY"] = "hispanic"
    df.loc[hispanic_mask, "ETHNICITY"] = "nonhispanic"


# === 4. 优化医疗计数指标的微聚合 ===
def micro_aggregate_medical_counts(df: pd.DataFrame) -> None:
    """降低医疗计数指标的扰动强度"""
    medical_ranges = {
        "encounter_count": (4, 1211),
        "num_procedures": (0, 500),
        "num_medications": (0, 200),
        "num_immunizations": (0, 50),
        "num_allergies": (0, 20),
        "num_devices": (0, 10)
    }

    medical_cols = list(medical_ranges.keys())

    # 确保年龄没有空值
    age_num = pd.to_numeric(df["AGE"], errors="coerce")
    empty_age_mask = age_num.isna()
    if empty_age_mask.any():
        median_age = age_num.median()
        if pd.isna(median_age):
            median_age = 45
        df.loc[empty_age_mask, "AGE"] = str(int(median_age))
        age_num = pd.to_numeric(df["AGE"], errors="coerce")

    for col in medical_cols:
        if col not in df.columns:
            continue

        min_val, max_val = medical_ranges[col]

        # 使用适中的年龄分组
        age_bins = [0, 18, 45, 65, 200]
        age_labels = ['0-17', '18-44', '45-64', '65+']

        df_temp = df.copy()
        df_temp['age_group_temp'] = pd.cut(age_num, bins=age_bins, labels=age_labels, right=False)

        for age_group in age_labels:
            group_mask = df_temp['age_group_temp'] == age_group
            if group_mask.sum() == 0:
                continue

            group_values = pd.to_numeric(df.loc[group_mask, col], errors="coerce")
            valid_mask = group_values.notna()

            if not valid_mask.any():
                continue

            valid_values = group_values[valid_mask]
            mean_val = valid_values.mean()
            std_val = valid_values.std()

            if std_val == 0 or pd.isna(std_val):
                std_val = max(1, mean_val * 0.1)

            # 降低噪声强度
            noise_std = std_val * 0.04  # 降低到4%
            noise = np.random.normal(0, noise_std, size=valid_mask.sum())

            new_values = (valid_values + noise).clip(lower=min_val, upper=max_val).round(0).astype(int)
            df.loc[group_mask & valid_mask, col] = new_values.astype(str)

        # 处理异常值
        current_values = pd.to_numeric(df[col], errors="coerce")
        out_of_range_mask = (current_values < min_val) | (current_values > max_val)
        if out_of_range_mask.any():
            median_val = current_values.median()
            if pd.isna(median_val):
                median_val = min_val
            replacement_val = int(np.clip(median_val, min_val, max_val))
            df.loc[out_of_range_mask, col] = str(replacement_val)

        empty_mask = (df[col] == "") | df[col].isna()
        if empty_mask.any():
            median_val = current_values.median()
            if pd.isna(median_val):
                median_val = min_val
            replacement_val = int(np.clip(median_val, min_val, max_val))
            df.loc[empty_mask, col] = str(replacement_val)


# === 5. 优化疾病标志位的条件扰动 ===
def conditional_flag_perturbation(df: pd.DataFrame) -> None:
    """降低疾病标志位的扰动强度，特别是asthma_flag"""
    if "AGE" not in df.columns:
        return

    age_num = pd.to_numeric(df["AGE"], errors="coerce")

    # asthma_flag: 降低扰动强度
    if "asthma_flag" in df.columns:
        # 温和的基于年龄的定向扰动
        young_asthma_mask = (age_num < 18) & (df["asthma_flag"] == "0")
        elderly_no_asthma_mask = (age_num > 60) & (df["asthma_flag"] == "1")

        # 温和增加儿童哮喘病例 (5%)
        if young_asthma_mask.any():
            young_add = df[young_asthma_mask].sample(frac=0.05, random_state=42).index
            df.loc[young_add, "asthma_flag"] = "1"

        # 温和减少老年人哮喘病例 (6%)
        if elderly_no_asthma_mask.any():
            elderly_remove = df[elderly_no_asthma_mask].sample(frac=0.06, random_state=42).index
            df.loc[elderly_remove, "asthma_flag"] = "0"

        # 降低全局随机扰动 (2%)
        global_flip_mask = (df["asthma_flag"].isin(["0", "1"])) & (np.random.rand(len(df)) < 0.02)
        df.loc[global_flip_mask & (df["asthma_flag"] == "0"), "asthma_flag"] = "1"
        df.loc[global_flip_mask & (df["asthma_flag"] == "1"), "asthma_flag"] = "0"

    # stroke_flag: 降低扰动
    if "stroke_flag" in df.columns:
        elderly_stroke_mask = (age_num > 65) & (df["stroke_flag"] == "1")

        if elderly_stroke_mask.any():
            # 4%的老年人中风改为无中风
            elderly_flip = df[elderly_stroke_mask].sample(frac=0.04, random_state=42).index
            df.loc[elderly_flip, "stroke_flag"] = "0"

    # obesity_flag: 降低扰动
    if "obesity_flag" in df.columns:
        middle_aged_obese = (age_num >= 40) & (age_num <= 60) & (df["obesity_flag"] == "0")
        young_obese = (age_num < 30) & (df["obesity_flag"] == "1")

        if middle_aged_obese.any():
            # 5%的中年无肥胖改为有肥胖
            middle_flip_indices = df[middle_aged_obese].sample(frac=0.05, random_state=42).index
            df.loc[middle_flip_indices, "obesity_flag"] = "1"

        if young_obese.any():
            # 6%的年轻有肥胖改为无肥胖
            young_obese_flip = df[young_obese].sample(frac=0.06, random_state=42).index
            df.loc[young_obese_flip, "obesity_flag"] = "0"

    # depression_flag: 温和调整患病率
    if "depression_flag" in df.columns:
        current_depression_rate = (df["depression_flag"] == "1").mean()
        target_depression_rate = 0.025  # 降低目标患病率

        if current_depression_rate < target_depression_rate:
            n_total = len(df)
            n_target_depressed = int(n_total * target_depression_rate)
            n_current_depressed = (df["depression_flag"] == "1").sum()
            n_to_add = n_target_depressed - n_current_depressed

            if n_to_add > 0:
                # 优先选择年轻人
                young_adult_mask = (age_num >= 18) & (age_num <= 40) & (df["depression_flag"] == "0")

                if young_adult_mask.any():
                    add_indices = df[young_adult_mask].sample(n=min(n_to_add, young_adult_mask.sum()),
                                                              random_state=42).index
                    df.loc[add_indices, "depression_flag"] = "1"


# === 6. 优化生理指标的智能扰动 ===
def smart_physiological_perturbation(df: pd.DataFrame) -> None:
    """降低生理指标的扰动强度"""

    physiological_ranges = {
        "mean_systolic_bp": (60, 200),
        "mean_diastolic_bp": (40, 120),
        "mean_weight": (30, 200),
        "mean_bmi": (15, 50)
    }

    # 血压指标：降低扰动
    for bp_col in ["mean_systolic_bp", "mean_diastolic_bp"]:
        if bp_col in df.columns:
            bp_values = pd.to_numeric(df[bp_col], errors="coerce")
            valid_mask = bp_values.notna()
            min_val, max_val = physiological_ranges[bp_col]

            if valid_mask.any():
                valid_bp = bp_values[valid_mask]

                # 降低噪声强度
                if bp_col == "mean_systolic_bp":
                    normal_noise_std = 1.5
                    adjustment_std = 2.0
                else:
                    normal_noise_std = 1.0
                    adjustment_std = 1.5

                # 正常范围扰动
                if bp_col == "mean_systolic_bp":
                    normal_bp_mask = (valid_bp >= 90) & (valid_bp <= 140)
                else:
                    normal_bp_mask = (valid_bp >= 60) & (valid_bp <= 90)

                abnormal_bp_mask = ~normal_bp_mask

                normal_noise = np.random.normal(0, normal_noise_std, normal_bp_mask.sum())
                valid_bp.loc[normal_bp_mask] = (valid_bp[normal_bp_mask] + normal_noise).clip(
                    physiological_ranges[bp_col][0], physiological_ranges[bp_col][1])

                # 异常值温和调整
                if abnormal_bp_mask.any():
                    adjustment = np.random.normal(2, adjustment_std, abnormal_bp_mask.sum())
                    if bp_col == "mean_systolic_bp":
                        high_bp_mask = abnormal_bp_mask & (valid_bp > 140)
                        low_bp_mask = abnormal_bp_mask & (valid_bp < 90)
                    else:
                        high_bp_mask = abnormal_bp_mask & (valid_bp > 90)
                        low_bp_mask = abnormal_bp_mask & (valid_bp < 60)

                    if high_bp_mask.any():
                        valid_bp.loc[high_bp_mask] = (
                                valid_bp[high_bp_mask] - np.abs(adjustment[:high_bp_mask.sum()])).clip(140, max_val)
                    if low_bp_mask.any():
                        valid_bp.loc[low_bp_mask] = (
                                    valid_bp[low_bp_mask] + np.abs(adjustment[:low_bp_mask.sum()])).clip(min_val, 90)

                valid_bp = valid_bp.clip(min_val, max_val)
                df.loc[valid_mask, bp_col] = valid_bp.round(0).astype(int).astype(str)

    # 体重和BMI：降低扰动
    for col in ["mean_weight", "mean_bmi"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            valid_mask = values.notna()
            min_val, max_val = physiological_ranges[col]

            if valid_mask.any():
                valid_values = values[valid_mask]

                if col == "mean_weight":
                    noise_std = 0.6  # 降低噪声
                else:
                    noise_std = 0.2

                noise = np.random.normal(0, noise_std, valid_mask.sum())
                new_values = (valid_values + noise).clip(min_val, max_val)

                if col == "mean_weight":
                    df.loc[valid_mask, col] = new_values.round(0).astype(int).astype(str)
                else:
                    df.loc[valid_mask, col] = new_values.round(1).astype(str)


# === 7. 优化性别比例微调 ===
def adjust_gender_ratio(df: pd.DataFrame) -> None:
    """降低性别扰动的强度"""
    if "GENDER" not in df.columns:
        return

    current_male_ratio = (df["GENDER"] == "M").mean()
    target_male_ratio = 0.5116

    n_total = len(df)
    n_target_male = int(n_total * target_male_ratio)
    n_current_male = (df["GENDER"] == "M").sum()

    n_change = abs(n_target_male - n_current_male)

    # 只进行必要的调整，不添加额外扰动
    if n_current_male > n_target_male:
        male_indices = df[df["GENDER"] == "M"].index
        if len(male_indices) > 0:
            change_indices = np.random.choice(male_indices, size=min(n_change, len(male_indices)), replace=False)
            df.loc[change_indices, "GENDER"] = "F"
    else:
        female_indices = df[df["GENDER"] == "F"].index
        if len(female_indices) > 0:
            change_indices = np.random.choice(female_indices, size=min(n_change, len(female_indices)), replace=False)
            df.loc[change_indices, "GENDER"] = "M"


# === 8. 优化列随机化 ===
def randomize_selected_columns(df: pd.DataFrame) -> None:
    """降低列随机化强度"""

    # 只对影响较小的列进行随机化
    columns_to_randomize = [
        "num_immunizations", "num_allergies", "num_devices"
    ]

    for col in columns_to_randomize:
        if col in df.columns:
            current_values = pd.to_numeric(df[col], errors="coerce")
            valid_mask = current_values.notna()

            if valid_mask.any():
                valid_values = current_values[valid_mask]

                # 降低随机化比例到10%
                change_mask = (np.random.rand(valid_mask.sum()) < 0.10)

                if change_mask.any():
                    change_values = valid_values[change_mask]

                    # 小范围变化：±1
                    random_changes = np.random.choice([-1, 0, 1],
                                                      size=change_mask.sum(),
                                                      p=[0.3, 0.4, 0.3])

                    new_values = (change_values + random_changes).clip(0, None)

                    change_indices = valid_values[change_mask].index
                    df.loc[change_indices, col] = new_values.astype(int).astype(str)


# === 9. 新增有用性保护策略 ===
def preserve_utility_features(df: pd.DataFrame) -> None:
    """保护关键有用性特征"""

    # 保护重要的统计关系
    if "AGE" in df.columns and "encounter_count" in df.columns:
        # 保持年龄与就诊次数的正相关关系
        age_num = pd.to_numeric(df["AGE"], errors="coerce")
        encounter_num = pd.to_numeric(df["encounter_count"], errors="coerce")

        # 计算原始相关系数
        original_corr = age_num.corr(encounter_num)

        # 如果相关系数变化太大，进行微调
        if abs(original_corr) < 0.1:  # 如果相关性变得太弱
            # 轻微增强年龄与就诊次数的关系
            age_encounter_mask = (age_num > 60) & (encounter_num < age_num.median())
            if age_encounter_mask.any():
                adjustment = np.random.randint(1, 3, age_encounter_mask.sum())
                new_encounters = (encounter_num[age_encounter_mask] + adjustment).clip(4, 1211)
                df.loc[age_encounter_mask, "encounter_count"] = new_encounters.astype(int).astype(str)


# === 10. 数据格式清理 ===
def clean_data_format(df: pd.DataFrame) -> None:
    """确保数据符合CiDataFrame的格式要求"""

    if "AGE" in df.columns:
        age_num = pd.to_numeric(df["AGE"], errors="coerce")
        empty_age_mask = age_num.isna()
        if empty_age_mask.any():
            median_age = age_num.median()
            if pd.isna(median_age):
                median_age = 45
            df.loc[empty_age_mask, "AGE"] = str(int(median_age))

    numeric_cols = [
        "AGE", "encounter_count", "num_procedures", "num_medications",
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
            df[col] = numeric_series.apply(
                lambda x: str(int(round(median_val))) if pd.isna(x) else str(int(round(x))))

    for col in float_cols:
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            median_val = numeric_series.median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = numeric_series.apply(
                lambda x: f"{median_val:.1f}" if pd.isna(x) else f"{x:.1f}")

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


def main():
    parser = argparse.ArgumentParser(description="优化版匿名化：平衡匿名性和有用性")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（再現用）")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    df = BiDataFrame.read_csv(args.input_csv)
    print(f"原始数据行数: {len(df)}")
    print("开始应用优化匿名化...")

    # 执行所有匿名化步骤（按有用性影响排序）
    steps = [
        ("数据格式清理", clean_data_format),
        ("种族合并", merge_race_to_other),
        ("民族扰动", safe_ethnicity_perturbation),
        ("年龄组互换", swap_age_groups),
        ("疾病标志位扰动", conditional_flag_perturbation),
        ("列随机化", randomize_selected_columns),
        ("医疗计数微聚合", micro_aggregate_medical_counts),
        ("生理指标扰动", smart_physiological_perturbation),
        ("性别比例调整", adjust_gender_ratio),
        ("有用性保护", preserve_utility_features),
        ("最终数据清理", clean_data_format)
    ]

    for step_name, step_func in steps:
        print(f"执行 {step_name}...")
        step_func(df)

    Ci_df = CiDataFrame(df)
    Ci_df.to_csv(args.output_csv)
    print(f"✅ 优化匿名化完成！输出文件: {args.output_csv}")
    print(f"处理后数据行数: {len(df)}")


if __name__ == "__main__":
    main()