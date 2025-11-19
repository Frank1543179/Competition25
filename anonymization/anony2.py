#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 优化版匿名化：在保持有用性70%+的前提下最小化统计差异
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


# === 1. 轻微年龄扰动 ===
def slight_age_perturbation(df: pd.DataFrame) -> None:
    """对年龄进行轻微扰动，保持年龄分布，范围限定在2~110"""
    if "AGE" not in df.columns:
        return

    # 转换为数值，处理空值
    age_num = pd.to_numeric(df["AGE"], errors="coerce")
    empty_age_mask = age_num.isna()

    if empty_age_mask.any():
        median_age = age_num.median()
        if pd.isna(median_age):
            median_age = 45
        df.loc[empty_age_mask, "AGE"] = str(int(median_age))

    # 重新获取年龄数值
    age_num = pd.to_numeric(df["AGE"], errors="coerce")

    # 约束在 [2, 110] 范围内
    age_num = age_num.clip(2, 110).astype(int)

    # 对5%的年龄进行±1岁的扰动
    perturb_mask = np.random.rand(len(df)) < 0.05
    perturb_direction = np.random.choice([-1, 1], size=perturb_mask.sum())

    new_ages = age_num[perturb_mask] + perturb_direction
    # 约束扰动后的年龄范围
    new_ages = new_ages.clip(2, 110).astype(int)

    # 更新扰动后的年龄
    df.loc[perturb_mask, "AGE"] = new_ages.astype(str)

    # 最终再统一保证范围正确且为整数
    final_age = pd.to_numeric(df["AGE"], errors="coerce").clip(2, 110).astype(int)
    df["AGE"] = final_age.astype(str)



# === 2. 种族合并为other ===
def merge_race_to_other(df: pd.DataFrame) -> None:
    """将hawaiian、native合并为other"""
    if "RACE" not in df.columns:
        return

    races_to_merge = ['hawaiian', 'native']
    mask = df["RACE"].isin(races_to_merge)
    df.loc[mask, "RACE"] = "other"

    print(f"种族合并完成：将 {mask.sum()} 条记录的种族合并为 'other'")


# === 3. 民族轻微扰动 ===
def slight_ethnicity_perturbation(df: pd.DataFrame) -> None:
    """对民族进行轻微扰动"""
    if "ETHNICITY" not in df.columns:
        return

    # 将少量nonhispanic改为hispanic（2%），hispanic改为nonhispanic（1%）
    nonhispanic_mask = (df["ETHNICITY"] == "nonhispanic") & (np.random.rand(len(df)) < 0.02)
    hispanic_mask = (df["ETHNICITY"] == "hispanic") & (np.random.rand(len(df)) < 0.01)

    df.loc[nonhispanic_mask, "ETHNICITY"] = "hispanic"
    df.loc[hispanic_mask, "ETHNICITY"] = "nonhispanic"


# === 4. 医疗计数指标的保守扰动 ===
def conservative_medical_counts_perturbation(df: pd.DataFrame) -> None:
    """对医疗计数指标进行保守扰动，保持分布特征"""
    medical_ranges = {
        "encounter_count": (4, 1211),
        "num_procedures": (0, 500),
        "num_medications": (0, 200),
        "num_immunizations": (0, 50),
        "num_allergies": (0, 20),
        "num_devices": (0, 10)
    }

    medical_cols = list(medical_ranges.keys())

    for col in medical_cols:
        if col not in df.columns:
            continue

        min_val, max_val = medical_ranges[col]
        current_values = pd.to_numeric(df[col], errors="coerce")

        # 只对3%的记录进行轻微扰动
        perturb_mask = (np.random.rand(len(df)) < 0.03) & current_values.notna()

        if perturb_mask.any():
            perturb_values = current_values[perturb_mask]
            # 使用很小的噪声 (±1-2)
            noise = np.random.choice([-2, -1, 1, 2], size=perturb_mask.sum())
            new_values = (perturb_values + noise).clip(min_val, max_val).astype(int)
            df.loc[perturb_mask, col] = new_values.astype(str)

        # 处理空值和超出范围的值
        empty_mask = (df[col] == "") | df[col].isna()
        out_of_range_mask = (current_values < min_val) | (current_values > max_val)

        if empty_mask.any() or out_of_range_mask.any():
            median_val = current_values.median()
            if pd.isna(median_val):
                median_val = min_val
            replacement_val = int(np.clip(median_val, min_val, max_val))
            df.loc[empty_mask | out_of_range_mask, col] = str(replacement_val)


# === 5. 疾病标志位的保守扰动 ===
def conservative_flag_perturbation(df: pd.DataFrame) -> None:
    """对疾病标志位进行保守扰动"""

    # asthma_flag: 轻微扰动
    if "asthma_flag" in df.columns:
        flip_mask = (df["asthma_flag"].isin(["0", "1"])) & (np.random.rand(len(df)) < 0.01)
        df.loc[flip_mask & (df["asthma_flag"] == "0"), "asthma_flag"] = "1"
        df.loc[flip_mask & (df["asthma_flag"] == "1"), "asthma_flag"] = "0"

    # depression_flag: 调整到合理患病率
    if "depression_flag" in df.columns:
        current_rate = (df["depression_flag"] == "1").mean()
        target_rate = 0.02  # 8%患病率

        if current_rate < target_rate:
            n_total = len(df)
            n_target = int(n_total * target_rate)
            n_current = (df["depression_flag"] == "1").sum()
            n_to_add = n_target - n_current

            if n_to_add > 0:
                # 优先选择年轻人增加抑郁症
                age_num = pd.to_numeric(df["AGE"], errors="coerce")
                young_mask = (age_num >= 18) & (age_num <= 40) & (df["depression_flag"] == "0")
                young_indices = df[young_mask].index.tolist()

                if len(young_indices) >= n_to_add:
                    add_indices = random.sample(young_indices, n_to_add)
                else:
                    add_indices = young_indices
                    remaining = n_to_add - len(young_indices)
                    # 从所有非抑郁症患者中选择
                    all_non_depressed = df[df["depression_flag"] == "0"].index.tolist()
                    remaining_candidates = list(set(all_non_depressed) - set(add_indices))
                    if len(remaining_candidates) >= remaining:
                        add_indices.extend(random.sample(remaining_candidates, remaining))

                df.loc[add_indices, "depression_flag"] = "1"
                print(f"增加了 {len(add_indices)} 个抑郁症病例")

    # stroke_flag和obesity_flag: 基于年龄的轻微扰动
    if "AGE" in df.columns:
        age_num = pd.to_numeric(df["AGE"], errors="coerce")

        # stroke_flag: 高龄轻微增加
        if "stroke_flag" in df.columns:
            old_no_stroke_mask = (age_num > 70) & (df["stroke_flag"] == "0") & (np.random.rand(len(df)) < 0.01)
            df.loc[old_no_stroke_mask, "stroke_flag"] = "1"

        # obesity_flag: 中年轻微增加
        if "obesity_flag" in df.columns:
            middle_no_obese_mask = (age_num >= 40) & (age_num <= 60) & (df["obesity_flag"] == "0") & (
                        np.random.rand(len(df)) < 0.01)
            df.loc[middle_no_obese_mask, "obesity_flag"] = "1"


# === 6. 生理指标的保守扰动 ===
def conservative_physiological_perturbation(df: pd.DataFrame) -> None:
    """对生理指标进行保守扰动"""

    physiological_ranges = {
        "mean_systolic_bp": (60, 200),
        "mean_diastolic_bp": (40, 120),
        "mean_weight": (30, 200),
        "mean_bmi": (15, 50)
    }

    # 血压：轻微扰动
    if "mean_systolic_bp" in df.columns:
        bp_values = pd.to_numeric(df["mean_systolic_bp"], errors="coerce")
        valid_mask = bp_values.notna()
        min_val, max_val = physiological_ranges["mean_systolic_bp"]

        if valid_mask.any():
            valid_bp = bp_values[valid_mask]
            # 对5%的记录进行±1 mmHg的扰动
            perturb_mask = np.random.rand(valid_mask.sum()) < 0.05
            noise = np.random.choice([-1, 1], size=perturb_mask.sum())
            valid_bp.loc[perturb_mask] = (valid_bp[perturb_mask] + noise).clip(min_val, max_val)
            df.loc[valid_mask, "mean_systolic_bp"] = valid_bp.round(0).astype(int).astype(str)

    # 体重：轻微扰动
    if "mean_weight" in df.columns:
        weight_values = pd.to_numeric(df["mean_weight"], errors="coerce")
        weight_valid_mask = weight_values.notna()
        min_val, max_val = physiological_ranges["mean_weight"]

        if weight_valid_mask.any():
            valid_weight = weight_values[weight_valid_mask]
            # 对3%的记录进行±0.5 kg的扰动
            perturb_mask = np.random.rand(weight_valid_mask.sum()) < 0.03
            noise = np.random.choice([-0.5, 0.5], size=perturb_mask.sum())
            new_weight = (valid_weight[perturb_mask] + noise).clip(min_val, max_val)
            df.loc[weight_valid_mask & perturb_mask, "mean_weight"] = new_weight.round(1).astype(str)

    # BMI：轻微扰动
    if "mean_bmi" in df.columns:
        bmi_values = pd.to_numeric(df["mean_bmi"], errors="coerce")
        bmi_valid_mask = bmi_values.notna()
        min_val, max_val = physiological_ranges["mean_bmi"]

        if bmi_valid_mask.any():
            valid_bmi = bmi_values[bmi_valid_mask]
            # 对3%的记录进行±0.1的扰动
            perturb_mask = np.random.rand(bmi_valid_mask.sum()) < 0.03
            noise = np.random.choice([-0.1, 0.1], size=perturb_mask.sum())
            new_bmi = (valid_bmi[perturb_mask] + noise).clip(min_val, max_val)
            df.loc[bmi_valid_mask & perturb_mask, "mean_bmi"] = new_bmi.round(1).astype(str)


# === 7. 性别比例微调 ===
def adjust_gender_ratio(df: pd.DataFrame) -> None:
    """微调性别比例"""
    if "GENDER" not in df.columns:
        return

    current_male_ratio = (df["GENDER"] == "M").mean()
    target_male_ratio = 0.5116

    n_total = len(df)
    n_target_male = int(n_total * target_male_ratio)
    n_current_male = (df["GENDER"] == "M").sum()

    n_change = abs(n_target_male - n_current_male)

    if n_change == 0:
        return

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


# === 8. 数据格式清理 ===
def clean_data_format(df: pd.DataFrame) -> None:
    """确保数据符合CiDataFrame的格式要求"""

    # 处理年龄
    if "AGE" in df.columns:
        age_num = pd.to_numeric(df["AGE"], errors="coerce")

        # 缺失值用中位数填充
        empty_age_mask = age_num.isna()
        if empty_age_mask.any():
            median_age = age_num.median()
            if pd.isna(median_age):
                median_age = 45
            df.loc[empty_age_mask, "AGE"] = str(int(round(median_age)))

        # 重新取整并强制范围 [2, 110]
        age_num = pd.to_numeric(df["AGE"], errors="coerce").round().astype(int)
        age_num = age_num.clip(lower=2, upper=110)

        df["AGE"] = age_num.astype(str)

    # 数值列处理（其他不变）
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
            df[col] = numeric_series.apply(
                lambda x: str(int(round(median_val))) if pd.isna(x) else str(int(round(x)))
            )

    for col in float_cols:
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            median_val = numeric_series.median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = numeric_series.apply(
                lambda x: f"{median_val:.1f}" if pd.isna(x) else f"{x:.1f}"
            )

    # 分类变量、标志变量的清理逻辑保持不变


    # 分类变量处理
    categorical_cols = ["GENDER", "RACE", "ETHNICITY"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": "", "None": "", "": ""})
            if df[col].isna().any() or (df[col] == "").any():
                most_common = df[col].mode()
                if len(most_common) > 0:
                    fill_value = most_common.iloc[0]
                    df[col] = df[col].replace({"": fill_value}).fillna(fill_value)

    # 标志变量处理
    flag_cols = ["asthma_flag", "stroke_flag", "obesity_flag", "depression_flag"]
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": "", "None": "", "": ""})
            valid_mask = df[col].isin(["0", "1"])
            df.loc[~valid_mask, col] = "0"


def main():
    parser = argparse.ArgumentParser(description="优化版匿名化：最小化统计差异，提高有用性")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（再現用）")
    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 读取数据
    df = BiDataFrame.read_csv(args.input_csv)
    print(f"原始数据行数: {len(df)}")

    # 应用优化匿名化方法
    print("开始应用优化匿名化...")

    # 1. 年龄轻微扰动
    print("执行年龄轻微扰动...")
    slight_age_perturbation(df)

    # 2. 种族合并
    print("执行种族合并...")
    merge_race_to_other(df)

    # 3. 民族轻微扰动
    print("执行民族轻微扰动...")
    slight_ethnicity_perturbation(df)

    # 4. 医疗计数保守扰动
    print("执行医疗计数保守扰动...")
    conservative_medical_counts_perturbation(df)

    # 5. 疾病标志位保守扰动
    print("执行疾病标志位保守扰动...")
    conservative_flag_perturbation(df)

    # 6. 生理指标保守扰动
    print("执行生理指标保守扰动...")
    conservative_physiological_perturbation(df)

    # 7. 性别比例微调
    print("执行性别比例微调...")
    adjust_gender_ratio(df)

    # 8. 数据格式清理
    print("执行数据格式清理...")
    clean_data_format(df)

    # 输出结果
    Ci_df = CiDataFrame(df)
    Ci_df.to_csv(args.output_csv)
    print(f"✅ 优化匿名化完成！输出文件: {args.output_csv}")
    print(f"处理后数据行数: {len(df)}")

    # 显示统计信息
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