#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 智能匿名化：基于数据特征的保护性匿名
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

# 年龄范围约束
AGE_MIN = 2
AGE_MAX = 110


# === 1. 基于分布的年龄扰动 ===
def distribution_based_age_perturbation(df: pd.DataFrame) -> None:
    """基于原始年龄分布进行智能扰动，约束在[2, 110]范围内"""
    if "AGE" not in df.columns:
        return

    age_num = pd.to_numeric(df["AGE"], errors="coerce")

    # 确保年龄在有效范围内
    age_num = age_num.clip(AGE_MIN, AGE_MAX)
    df["AGE"] = age_num.astype(int).astype(str)

    # 基于原始分布创建扰动
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
    original_dist = pd.cut(age_num, bins=age_bins).value_counts(normalize=True)

    # 轻微调整年龄分布，保持整体模式
    for i in range(len(age_bins) - 1):
        age_mask = (age_num >= age_bins[i]) & (age_num < age_bins[i + 1])
        if age_mask.sum() > 10:  # 只在有足够样本的组内扰动
            # 轻微扰动：±2岁，确保在[2, 110]范围内
            perturbation = np.random.randint(-2, 3, age_mask.sum())
            new_ages = (age_num[age_mask] + perturbation).clip(AGE_MIN, AGE_MAX)
            df.loc[age_mask, "AGE"] = new_ages.astype(int).astype(str)


# === 2. 保护性种族匿名化 ===
def protective_race_anonymization(df: pd.DataFrame) -> None:
    """保护主要种族分布，合并极少数种族"""
    if "RACE" not in df.columns:
        return

    # 只合并数量最少的种族（<2%）
    rare_races = ['hawaiian', 'native', 'other', 'asian', 'black']

    for race in rare_races:
        if race in df["RACE"].values:
            # 将稀有种族随机分配到主要种族中
            race_mask = df["RACE"] == race
            if race_mask.sum() > 0:
                # 主要保持white，少量分配到其他
                new_races = np.random.choice(['white', 'other'],
                                             size=race_mask.sum(),
                                             p=[0.8, 0.2])
                df.loc[race_mask, "RACE"] = new_races


# === 3. 民族比例保持扰动 ===
def ethnicity_proportion_preserving_perturbation(df: pd.DataFrame) -> None:
    """保持民族比例的同时进行轻微扰动"""
    if "ETHNICITY" not in df.columns:
        return

    # 计算当前比例
    current_hispanic = (df["ETHNICITY"] == "hispanic").mean()
    target_hispanic = 0.0655  # 基于你的数据

    # 轻微调整以接近目标比例
    if current_hispanic < target_hispanic:
        # 需要增加hispanic
        n_to_change = int(len(df) * (target_hispanic - current_hispanic) * 0.5)
        nonhispanic_indices = df[df["ETHNICITY"] == "nonhispanic"].index
        if len(nonhispanic_indices) > n_to_change:
            change_indices = np.random.choice(nonhispanic_indices, n_to_change, replace=False)
            df.loc[change_indices, "ETHNICITY"] = "hispanic"
    else:
        # 需要减少hispanic
        n_to_change = int(len(df) * (current_hispanic - target_hispanic) * 0.5)
        hispanic_indices = df[df["ETHNICITY"] == "hispanic"].index
        if len(hispanic_indices) > n_to_change:
            change_indices = np.random.choice(hispanic_indices, n_to_change, replace=False)
            df.loc[change_indices, "ETHNICITY"] = "nonhispanic"


# === 4. 医疗计数分布保持扰动 ===
def medical_count_distribution_preserving(df: pd.DataFrame) -> None:
    """保持医疗计数分布特征的扰动"""

    medical_columns = {
        "encounter_count": (7, 85, 65.99),
        "num_procedures": (1, 2086, 181.21),
        "num_medications": (0, 2351, 57.80),
        "num_immunizations": (0, 37, 13.86),
        "num_allergies": (0, 13, 0.86),
        "num_devices": (0, 133, 6.53)
    }

    for col, (min_val, max_val, mean_val) in medical_columns.items():
        if col not in df.columns:
            continue

        current_values = pd.to_numeric(df[col], errors="coerce")

        # 基于当前值的相对扰动
        if col in ["encounter_count", "num_procedures", "num_medications"]:
            # 对高值列使用百分比扰动
            perturbation = np.random.normal(0, 0.02, len(df))  # 2%的扰动
            new_values = (current_values * (1 + perturbation)).clip(min_val, max_val).round(0)
        else:
            # 对低值列使用绝对扰动
            perturbation = np.random.randint(-1, 2, len(df))  # ±1的扰动
            new_values = (current_values + perturbation).clip(min_val, max_val)

        df[col] = new_values.astype(int).astype(str)


# === 5. 疾病标志位相关性保护扰动 ===
def disease_flag_correlation_preserving(df: pd.DataFrame) -> None:
    """保护疾病标志位与年龄的相关性，考虑年龄约束"""
    if "AGE" not in df.columns:
        return

    age_num = pd.to_numeric(df["AGE"], errors="coerce")

    # asthma_flag: 儿童和年轻人患病率较高
    if "asthma_flag" in df.columns:
        # 基于年龄的合理扰动
        young_asthma_mask = (age_num < 30) & (df["asthma_flag"] == "0")
        elderly_asthma_mask = (age_num > 60) & (df["asthma_flag"] == "1")

        # 轻微增加年轻人哮喘
        if young_asthma_mask.any():
            young_add = young_asthma_mask & (np.random.rand(len(df)) < 0.02)
            df.loc[young_add, "asthma_flag"] = "1"

        # 轻微减少老年人哮喘
        if elderly_asthma_mask.any():
            elderly_remove = elderly_asthma_mask & (np.random.rand(len(df)) < 0.01)
            df.loc[elderly_remove, "asthma_flag"] = "0"

    # stroke_flag: 老年人患病率较高
    if "stroke_flag" in df.columns:
        elderly_stroke_mask = (age_num > 65) & (df["stroke_flag"] == "0")
        young_stroke_mask = (age_num < 45) & (df["stroke_flag"] == "1")

        # 轻微增加老年人中风
        if elderly_stroke_mask.any():
            elderly_add = elderly_stroke_mask & (np.random.rand(len(df)) < 0.01)
            df.loc[elderly_add, "stroke_flag"] = "1"

        # 轻微减少年轻人中风
        if young_stroke_mask.any():
            young_remove = young_stroke_mask & (np.random.rand(len(df)) < 0.005)
            df.loc[young_remove, "stroke_flag"] = "0"

    # obesity_flag: 中年人患病率较高
    if "obesity_flag" in df.columns:
        middle_aged_obese = (age_num >= 40) & (age_num <= 60) & (df["obesity_flag"] == "0")
        young_obese = (age_num < 30) & (df["obesity_flag"] == "1")

        # 轻微增加中年人肥胖
        if middle_aged_obese.any():
            middle_add = middle_aged_obese & (np.random.rand(len(df)) < 0.01)
            df.loc[middle_add, "obesity_flag"] = "1"

        # 轻微减少年轻人肥胖
        if young_obese.any():
            young_remove = young_obese & (np.random.rand(len(df)) < 0.01)
            df.loc[young_remove, "obesity_flag"] = "0"

    # depression_flag: 轻微增加患病率到合理水平
    if "depression_flag" in df.columns:
        current_rate = (df["depression_flag"] == "1").mean()
        if current_rate < 0.02:  # 目标2%
            n_to_add = int(len(df) * (0.02 - current_rate))
            candidates = df[df["depression_flag"] == "0"].index
            if len(candidates) > n_to_add:
                add_indices = np.random.choice(candidates, n_to_add, replace=False)
                df.loc[add_indices, "depression_flag"] = "1"


# === 6. 生理指标范围保护扰动 ===
def physiological_range_preserving(df: pd.DataFrame) -> None:
    """保护生理指标的合理范围和相关性"""

    # 血压：保护正常/异常比例
    if "mean_systolic_bp" in df.columns:
        systolic_bp = pd.to_numeric(df["mean_systolic_bp"], errors="coerce")
        normal_mask = (systolic_bp >= 90) & (systolic_bp <= 140)

        # 对异常值进行轻微调整
        abnormal_mask = ~normal_mask
        if abnormal_mask.any():
            # 异常值向正常范围轻微调整
            adjustment = np.random.normal(5, 2, abnormal_mask.sum())
            high_bp_mask = abnormal_mask & (systolic_bp > 140)
            low_bp_mask = abnormal_mask & (systolic_bp < 90)

            if high_bp_mask.any():
                new_values = (systolic_bp[high_bp_mask] - np.abs(adjustment[:high_bp_mask.sum()])).clip(140, 200)
                df.loc[high_bp_mask, "mean_systolic_bp"] = new_values.round(0).astype(int).astype(str)

            if low_bp_mask.any():
                new_values = (systolic_bp[low_bp_mask] + np.abs(adjustment[:low_bp_mask.sum()])).clip(60, 90)
                df.loc[low_bp_mask, "mean_systolic_bp"] = new_values.round(0).astype(int).astype(str)

    # BMI：保护分布特征
    if "mean_bmi" in df.columns:
        bmi_values = pd.to_numeric(df["mean_bmi"], errors="coerce")
        # 轻微扰动，保持肥胖率
        perturbation = np.random.normal(0, 0.5, len(df))
        new_bmi = (bmi_values + perturbation).clip(15, 50)
        df["mean_bmi"] = new_bmi.round(1).astype(str)

    # 体重：保护正常范围
    if "mean_weight" in df.columns:
        weight_values = pd.to_numeric(df["mean_weight"], errors="coerce")
        # 轻微扰动
        perturbation = np.random.normal(0, 0.8, len(df))
        new_weight = (weight_values + perturbation).clip(30, 200)
        df["mean_weight"] = new_weight.round(0).astype(int).astype(str)


# === 7. 性别比例精确调整 ===
def precise_gender_adjustment(df: pd.DataFrame) -> None:
    """精确调整性别比例到目标值"""
    if "GENDER" not in df.columns:
        return

    current_male_ratio = (df["GENDER"] == "M").mean()
    target_male_ratio = 0.5116

    n_total = len(df)
    n_target_male = int(n_total * target_male_ratio)
    n_current_male = (df["GENDER"] == "M").sum()

    n_change = n_target_male - n_current_male

    if n_change > 0:
        # 需要增加男性
        female_indices = df[df["GENDER"] == "F"].index
        if len(female_indices) >= n_change:
            change_indices = np.random.choice(female_indices, n_change, replace=False)
            df.loc[change_indices, "GENDER"] = "M"
    elif n_change < 0:
        # 需要减少男性
        male_indices = df[df["GENDER"] == "M"].index
        if len(male_indices) >= abs(n_change):
            change_indices = np.random.choice(male_indices, abs(n_change), replace=False)
            df.loc[change_indices, "GENDER"] = "F"


# === 8. 微聚合保护有用性 ===
def micro_aggregation_utility_preserving(df: pd.DataFrame) -> None:
    """使用微聚合保护统计特征"""

    # 按年龄组进行微聚合
    if "AGE" in df.columns:
        age_num = pd.to_numeric(df["AGE"], errors="coerce")
        # 确保年龄在有效范围内
        age_num = age_num.clip(AGE_MIN, AGE_MAX)
        age_groups = pd.cut(age_num, bins=[0, 18, 35, 50, 65, 80, 200])

        # 对医疗计数进行组内微聚合
        medical_cols = ["encounter_count", "num_procedures", "num_medications",
                        "num_immunizations", "num_allergies", "num_devices"]

        for col in medical_cols:
            if col in df.columns:
                for group in age_groups.cat.categories:
                    group_mask = age_groups == group
                    if group_mask.sum() > 5:  # 只在有足够样本的组内操作
                        group_values = pd.to_numeric(df.loc[group_mask, col], errors="coerce")
                        if group_values.notna().any():
                            group_mean = group_values.mean()
                            # 组内值向组均值轻微靠拢
                            adjustment = (group_mean - group_values) * 0.1  # 10%的调整
                            new_values = (group_values + adjustment).clip(0, None).round(0)
                            df.loc[group_mask, col] = new_values.astype(int).astype(str)


# === 9. 记录交换匿名化 ===
def record_swapping_anonymization(df: pd.DataFrame) -> None:
    """交换相似记录的某些属性，考虑年龄约束"""

    n_swaps = int(len(df) * 0.02)  # 交换2%的记录

    if n_swaps > 1:
        # 基于年龄和性别找到相似记录
        age_num = pd.to_numeric(df["AGE"], errors="coerce")

        for _ in range(n_swaps):
            # 随机选择一个记录
            idx1 = np.random.randint(0, len(df))
            age1 = age_num.iloc[idx1]
            gender1 = df.iloc[idx1]["GENDER"]

            # 找到相似记录（相同性别，年龄相差±5岁）
            similar_mask = (df["GENDER"] == gender1) & (abs(age_num - age1) <= 5)
            similar_indices = df[similar_mask].index.tolist()

            if len(similar_indices) > 1:
                # 移除自己
                similar_indices = [idx for idx in similar_indices if idx != idx1]
                if similar_indices:
                    idx2 = np.random.choice(similar_indices)

                    # 交换非敏感属性
                    swap_columns = ["num_immunizations", "num_allergies", "num_devices"]
                    for col in swap_columns:
                        if col in df.columns:
                            temp = df.loc[idx1, col]
                            df.loc[idx1, col] = df.loc[idx2, col]
                            df.loc[idx2, col] = temp


# === 10. 年龄范围约束函数 ===
def enforce_age_constraints(df: pd.DataFrame) -> None:
    """确保所有年龄都在[2, 110]范围内"""
    if "AGE" in df.columns:
        age_num = pd.to_numeric(df["AGE"], errors="coerce")

        # 检查并修复超出范围的年龄
        out_of_range_mask = (age_num < AGE_MIN) | (age_num > AGE_MAX)
        if out_of_range_mask.any():
            print(f"发现 {out_of_range_mask.sum()} 个超出范围的年龄值，正在修复...")

            # 对于超出范围的年龄，使用该年龄组的均值
            age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
            age_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59',
                          '60-69', '70-79', '80-89', '90-99', '100+']

            df_temp = df.copy()
            df_temp['age_group'] = pd.cut(age_num, bins=age_bins, labels=age_labels, right=False)

            for age_group in age_labels:
                group_mask = (df_temp['age_group'] == age_group) & out_of_range_mask
                if group_mask.any():
                    # 使用该年龄组的均值，但约束在范围内
                    group_mean = age_num[df_temp['age_group'] == age_group].mean()
                    replacement_age = int(np.clip(group_mean, AGE_MIN, AGE_MAX))
                    df.loc[group_mask, "AGE"] = str(replacement_age)

            # 再次检查确保所有年龄都在范围内
            final_age_num = pd.to_numeric(df["AGE"], errors="coerce")
            final_out_of_range = (final_age_num < AGE_MIN) | (final_age_num > AGE_MAX)
            if final_out_of_range.any():
                # 如果还有超出范围的，使用中位数
                median_age = final_age_num.median()
                replacement_age = int(np.clip(median_age, AGE_MIN, AGE_MAX))
                df.loc[final_out_of_range, "AGE"] = str(replacement_age)


# === 11. 数据格式清理和验证 ===
def clean_and_validate_data(df: pd.DataFrame) -> None:
    """确保数据格式正确并验证有用性，包括年龄约束"""

    # 首先强制执行年龄约束
    enforce_age_constraints(df)

    # 处理年龄
    if "AGE" in df.columns:
        age_num = pd.to_numeric(df["AGE"], errors="coerce")
        empty_age_mask = age_num.isna()
        if empty_age_mask.any():
            median_age = age_num.median()
            replacement_age = int(np.clip(median_age, AGE_MIN, AGE_MAX))
            df.loc[empty_age_mask, "AGE"] = str(replacement_age)

    # 数值列格式化
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
            df[col] = numeric_series.fillna(median_val).round(0).astype(int).astype(str)

    for col in float_cols:
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            median_val = numeric_series.median()
            df[col] = numeric_series.fillna(median_val).round(1).astype(str)

    # 分类变量清理
    categorical_cols = ["GENDER", "RACE", "ETHNICITY"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": "", "None": ""})
            if df[col].isna().any() or (df[col] == "").any():
                most_common = df[col].mode()
                if len(most_common) > 0:
                    fill_value = most_common.iloc[0]
                    df[col] = df[col].fillna(fill_value).replace("", fill_value)

    # 标志变量清理
    flag_cols = ["asthma_flag", "stroke_flag", "obesity_flag", "depression_flag"]
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": "0", "None": "0", "": "0"})
            df[col] = df[col].apply(lambda x: "1" if x == "1" else "0")

    # 最终年龄范围验证
    if "AGE" in df.columns:
        final_age_check = pd.to_numeric(df["AGE"], errors="coerce")
        if (final_age_check < AGE_MIN).any() or (final_age_check > AGE_MAX).any():
            print("警告：仍有年龄值超出范围，进行最终修复...")
            final_age_check = final_age_check.clip(AGE_MIN, AGE_MAX)
            df["AGE"] = final_age_check.astype(int).astype(str)


def main():
    parser = argparse.ArgumentParser(description="智能匿名化：基于数据特征的保护性匿名")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（再現用）")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    df = BiDataFrame.read_csv(args.input_csv)
    print(f"原始数据行数: {len(df)}")
    print("开始应用智能匿名化...")

    # 执行匿名化步骤（按有用性影响从小到大）
    steps = [
        ("数据清理和验证", clean_and_validate_data),
        ("性别比例调整", precise_gender_adjustment),
        ("民族比例保护", ethnicity_proportion_preserving_perturbation),
        ("生理指标保护", physiological_range_preserving),
        ("疾病标志位保护", disease_flag_correlation_preserving),
        ("医疗计数分布保护", medical_count_distribution_preserving),
        ("微聚合保护", micro_aggregation_utility_preserving),
        ("种族匿名化", protective_race_anonymization),
        ("年龄分布扰动", distribution_based_age_perturbation),
        ("记录交换", record_swapping_anonymization),
        ("最终数据清理和年龄验证", clean_and_validate_data)
    ]

    for step_name, step_func in steps:
        print(f"执行 {step_name}...")
        step_func(df)

    # 最终年龄范围检查
    if "AGE" in df.columns:
        final_ages = pd.to_numeric(df["AGE"], errors="coerce")
        min_age = final_ages.min()
        max_age = final_ages.max()
        print(f"最终年龄范围: {min_age} - {max_age}")
        if min_age < AGE_MIN or max_age > AGE_MAX:
            print("警告：年龄范围超出约束！")

    Ci_df = CiDataFrame(df)
    Ci_df.to_csv(args.output_csv)
    print(f"✅ 智能匿名化完成！输出文件: {args.output_csv}")
    print(f"处理后数据行数: {len(df)}")


if __name__ == "__main__":
    main()