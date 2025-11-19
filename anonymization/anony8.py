#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 增强版匿名化：在保持有用性70%+的前提下提高匿名性（符合格式约束）
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


# === 1. 年龄组互换策略（修复空值问题）===
def swap_age_groups(df: pd.DataFrame) -> None:
    """对相似年龄组进行互换：21-30↔31-40, 51-60↔61-70"""
    if "AGE" not in df.columns:
        return

    # 确保年龄没有空值
    age_num = pd.to_numeric(df["AGE"], errors="coerce")
    empty_age_mask = age_num.isna()

    # 如果有空年龄，用中位数填充
    if empty_age_mask.any():
        median_age = age_num.median()
        if pd.isna(median_age):
            median_age = 45  # 默认值
        df.loc[empty_age_mask, "AGE"] = str(int(median_age))
        print(f"修复了 {empty_age_mask.sum()} 个空年龄值")

    age_num = pd.to_numeric(df["AGE"], errors="coerce")

    # 第一组互换：21-30岁 ↔ 31-40岁
    group1_young = (age_num >= 21) & (age_num <= 30)
    group1_old = (age_num >= 31) & (age_num <= 40)

    young_indices = df[group1_young].index.tolist()
    old_indices = df[group1_old].index.tolist()

    # 随机选择8%的记录进行互换（略微降低比例）
    n_swap = min(int(len(young_indices) * 0.08), int(len(old_indices) * 0.08))

    if n_swap > 0:
        swap_young = random.sample(young_indices, n_swap)
        swap_old = random.sample(old_indices, n_swap)

        # 互换年龄值
        young_ages = df.loc[swap_young, "AGE"].copy()
        old_ages = df.loc[swap_old, "AGE"].copy()

        df.loc[swap_young, "AGE"] = old_ages
        df.loc[swap_old, "AGE"] = young_ages

    # 第二组互换：51-60岁 ↔ 61-70岁
    group2_young = (age_num >= 51) & (age_num <= 60)
    group2_old = (age_num >= 61) & (age_num <= 70)

    young_indices2 = df[group2_young].index.tolist()
    old_indices2 = df[group2_old].index.tolist()

    n_swap2 = min(int(len(young_indices2) * 0.08), int(len(old_indices2) * 0.08))

    if n_swap2 > 0:
        swap_young2 = random.sample(young_indices2, n_swap2)
        swap_old2 = random.sample(old_indices2, n_swap2)

        young_ages2 = df.loc[swap_young2, "AGE"].copy()
        old_ages2 = df.loc[swap_old2, "AGE"].copy()

        df.loc[swap_young2, "AGE"] = old_ages2
        df.loc[swap_old2, "AGE"] = young_ages2


# === 2. 种族合并为other ===
def merge_race_to_other(df: pd.DataFrame) -> None:
    """将hawaiian、native合并为other"""
    if "RACE" not in df.columns:
        return

    # 要合并的种族类别：hawaiian和native
    races_to_merge = ['hawaiian', 'native']

    # 将这些种族的100%合并为"other"
    mask = df["RACE"].isin(races_to_merge)
    df.loc[mask, "RACE"] = "other"

    print(f"种族合并完成：将 {mask.sum()} 条记录的种族合并为 'other'")


# === 3. 民族安全扰动 ===
def safe_ethnicity_perturbation(df: pd.DataFrame) -> None:
    """在允许的民族值范围内进行扰动"""
    if "ETHNICITY" not in df.columns:
        return

    # 允许的民族值
    allowed_ethnicities = ['hispanic', 'nonhispanic', 'unknown']

    # 将少量nonhispanic改为hispanic（4%），hispanic改为nonhispanic（2%）
    nonhispanic_mask = (df["ETHNICITY"] == "nonhispanic") & (np.random.rand(len(df)) < 0.04)
    hispanic_mask = (df["ETHNICITY"] == "hispanic") & (np.random.rand(len(df)) < 0.02)

    df.loc[nonhispanic_mask, "ETHNICITY"] = "hispanic"
    df.loc[hispanic_mask, "ETHNICITY"] = "nonhispanic"


# === 4. 医疗计数指标的微聚合 ===
def micro_aggregate_medical_counts(df: pd.DataFrame) -> None:
    """对医疗计数指标进行微聚合加噪，保持整数格式且在允许范围内"""
    # 定义每个医疗计数指标的允许范围
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

        # 获取该列的允许范围
        min_val, max_val = medical_ranges[col]

        # 按新的年龄组分组建模
        age_bins = [0, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 200]
        age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '100+']

        df_temp = df.copy()
        df_temp['age_group_temp'] = pd.cut(age_num, bins=age_bins, labels=age_labels, right=False)

        # 对每个年龄组内的医疗计数进行微聚合
        for age_group in age_labels:
            group_mask = df_temp['age_group_temp'] == age_group
            if group_mask.sum() == 0:
                continue

            # 获取该年龄组的原始值
            group_values = pd.to_numeric(df.loc[group_mask, col], errors="coerce")
            valid_mask = group_values.notna()

            if not valid_mask.any():
                continue

            # 计算该年龄组的均值和标准差
            valid_values = group_values[valid_mask]
            mean_val = valid_values.mean()
            std_val = valid_values.std()

            if std_val == 0 or pd.isna(std_val):
                std_val = max(1, mean_val * 0.1)

            # 添加基于年龄组分布的噪声，保持整数且在允许范围内（降低噪声强度）
            noise = np.random.normal(0, std_val * 0.03, size=valid_mask.sum())
            new_values = (valid_values + noise).clip(lower=min_val, upper=max_val).round(0).astype(int)

            # 确保输出为字符串格式
            df.loc[group_mask & valid_mask, col] = new_values.astype(str)

        # 处理空值和超出范围的值
        current_values = pd.to_numeric(df[col], errors="coerce")
        # 将超出范围的值调整到允许范围内
        out_of_range_mask = (current_values < min_val) | (current_values > max_val)
        if out_of_range_mask.any():
            # 对于超出范围的值，使用该列的均值或中位数
            median_val = current_values.median()
            if pd.isna(median_val):
                median_val = min_val
            replacement_val = int(np.clip(median_val, min_val, max_val))
            df.loc[out_of_range_mask, col] = str(replacement_val)

        # 处理空值
        empty_mask = (df[col] == "") | df[col].isna()
        if empty_mask.any():
            median_val = current_values.median()
            if pd.isna(median_val):
                median_val = min_val
            replacement_val = int(np.clip(median_val, min_val, max_val))
            df.loc[empty_mask, col] = str(replacement_val)


# === 5. 疾病标志位的条件扰动（调整asthma_flag扰动策略）===
def conditional_flag_perturbation(df: pd.DataFrame) -> None:
    """基于年龄条件对疾病标志位进行扰动，保持0/1格式"""
    if "AGE" not in df.columns:
        return

    age_num = pd.to_numeric(df["AGE"], errors="coerce")

    # stroke_flag: 基于年龄的合理扰动
    old_stroke_mask = (age_num > 70) & (df["stroke_flag"] == "1")
    young_no_stroke_mask = (age_num < 40) & (df["stroke_flag"] == "0")

    # 选择1%的高龄中风患者改为无中风（降低比例）
    if old_stroke_mask.any():
        old_flip_indices = df[old_stroke_mask].sample(frac=0.01, random_state=42).index
        df.loc[old_flip_indices, "stroke_flag"] = "0"

    # 选择0.5%的低龄无中风改为有中风（降低比例）
    if young_no_stroke_mask.any():
        young_flip_indices = df[young_no_stroke_mask].sample(frac=0.005, random_state=42).index
        df.loc[young_flip_indices, "stroke_flag"] = "1"

    # obesity_flag: 年龄相关性扰动
    middle_aged_obese = (age_num >= 40) & (age_num <= 60) & (df["obesity_flag"] == "0")
    young_obese = (age_num < 30) & (df["obesity_flag"] == "1")

    # 中年无肥胖的2%改为有肥胖（降低比例）
    if middle_aged_obese.any():
        middle_flip_indices = df[middle_aged_obese].sample(frac=0.02, random_state=42).index
        df.loc[middle_flip_indices, "obesity_flag"] = "1"

    # 年轻有肥胖的3%改为无肥胖（降低比例）
    if young_obese.any():
        young_obese_flip = df[young_obese].sample(frac=0.03, random_state=42).index
        df.loc[young_obese_flip, "obesity_flag"] = "0"

    # asthma_flag: 降低扰动强度以控制LR_asthma_diff
    if "asthma_flag" in df.columns:
        # 对所有记录的1%进行随机翻转（降低扰动比例）
        flip_mask = (df["asthma_flag"].isin(["0", "1"])) & (np.random.rand(len(df)) < 0.01)
        df.loc[flip_mask & (df["asthma_flag"] == "0"), "asthma_flag"] = "1"
        df.loc[flip_mask & (df["asthma_flag"] == "1"), "asthma_flag"] = "0"

        # 额外策略：基于年龄的轻微调整，进一步降低LR差异
        # 儿童哮喘患者略微减少（0.5%）
        child_asthma_mask = (age_num < 18) & (df["asthma_flag"] == "1")
        if child_asthma_mask.any():
            child_reduce = df[child_asthma_mask].sample(frac=0.005, random_state=42).index
            df.loc[child_reduce, "asthma_flag"] = "0"

        # 成人非哮喘患者略微增加（0.3%）
        adult_no_asthma_mask = (age_num >= 18) & (age_num <= 40) & (df["asthma_flag"] == "0")
        if adult_no_asthma_mask.any():
            adult_increase = df[adult_no_asthma_mask].sample(frac=0.003, random_state=42).index
            df.loc[adult_increase, "asthma_flag"] = "1"

    # depression_flag: 增加患病率到约2%
    if "depression_flag" in df.columns:
        current_depression_rate = (df["depression_flag"] == "1").mean()
        target_depression_rate = 0.02  # 目标患病率2%

        if current_depression_rate < target_depression_rate:
            # 需要增加抑郁症患者
            n_total = len(df)
            n_target_depressed = int(n_total * target_depression_rate)
            n_current_depressed = (df["depression_flag"] == "1").sum()
            n_to_add = n_target_depressed - n_current_depressed

            if n_to_add > 0:
                # 优先选择年轻人（18-40岁）和女性增加抑郁症
                young_adult_mask = (age_num >= 18) & (age_num <= 40) & (df["depression_flag"] == "0")
                female_mask = (df["GENDER"] == "F") & (df["depression_flag"] == "0")

                # 结合条件：年轻女性优先
                priority_mask = young_adult_mask & female_mask
                secondary_mask = young_adult_mask | female_mask

                # 先选择年轻女性
                priority_indices = df[priority_mask].index.tolist()
                if len(priority_indices) >= n_to_add:
                    add_indices = random.sample(priority_indices, n_to_add)
                else:
                    add_indices = priority_indices
                    remaining = n_to_add - len(priority_indices)
                    # 再选择其他符合条件的
                    secondary_candidates = list(set(df[secondary_mask].index.tolist()) - set(priority_indices))
                    if len(secondary_candidates) >= remaining:
                        add_indices.extend(random.sample(secondary_candidates, remaining))
                    else:
                        # 如果还不够，从所有非抑郁症患者中选择
                        all_non_depressed = df[df["depression_flag"] == "0"].index.tolist()
                        remaining_candidates = list(set(all_non_depressed) - set(add_indices))
                        if len(remaining_candidates) >= remaining:
                            add_indices.extend(random.sample(remaining_candidates, remaining))

                df.loc[add_indices, "depression_flag"] = "1"
                print(f"增加了 {len(add_indices)} 个抑郁症病例")


# === 6. 生理指标的智能扰动 ===
def smart_physiological_perturbation(df: pd.DataFrame) -> None:
    """基于正常/异常比例对生理指标进行智能扰动，确保数值范围合理"""

    # 定义生理指标的允许范围
    physiological_ranges = {
        "mean_systolic_bp": (60, 200),
        "mean_diastolic_bp": (40, 120),
        "mean_weight": (30, 200),
        "mean_bmi": (15, 50)
    }

    # 血压指标：正常人群扰动较小，异常人群适当调整
    if "mean_systolic_bp" in df.columns:
        bp_values = pd.to_numeric(df["mean_systolic_bp"], errors="coerce")
        valid_mask = bp_values.notna()
        min_val, max_val = physiological_ranges["mean_systolic_bp"]

        if valid_mask.any():
            valid_bp = bp_values[valid_mask]
            # 正常血压范围：90-140 mmHg
            normal_bp_mask = (valid_bp >= 90) & (valid_bp <= 140)
            abnormal_bp_mask = ~normal_bp_mask

            # 对正常血压添加小噪声（降低噪声强度）
            normal_noise = np.random.normal(0, 1.0, normal_bp_mask.sum())
            valid_bp.loc[normal_bp_mask] = (valid_bp[normal_bp_mask] + normal_noise).clip(90, 140)

            # 对异常血压适当向正常范围调整（降低调整幅度）
            if abnormal_bp_mask.any():
                adjustment = np.random.normal(2, 1.5, abnormal_bp_mask.sum())
                # 高血压向下调，低血压向上调
                high_bp_mask = abnormal_bp_mask & (valid_bp > 140)
                low_bp_mask = abnormal_bp_mask & (valid_bp < 90)

                if high_bp_mask.any():
                    valid_bp.loc[high_bp_mask] = (
                            valid_bp[high_bp_mask] - np.abs(adjustment[:high_bp_mask.sum()])).clip(140, max_val)
                if low_bp_mask.any():
                    valid_bp.loc[low_bp_mask] = (valid_bp[low_bp_mask] + np.abs(adjustment[:low_bp_mask.sum()])).clip(
                        min_val, 90)

            # 确保在允许范围内
            valid_bp = valid_bp.clip(min_val, max_val)
            df.loc[valid_mask, "mean_systolic_bp"] = valid_bp.round(0).astype(int).astype(str)

    # 体重：简单小噪声扰动
    if "mean_weight" in df.columns:
        weight_values = pd.to_numeric(df["mean_weight"], errors="coerce")
        weight_valid_mask = weight_values.notna()
        min_val, max_val = physiological_ranges["mean_weight"]

        if weight_valid_mask.any():
            valid_weight = weight_values[weight_valid_mask]
            weight_noise = np.random.normal(0, 0.5, weight_valid_mask.sum())
            new_weight = (valid_weight + weight_noise).clip(min_val, max_val)
            df.loc[weight_valid_mask, "mean_weight"] = new_weight.round(0).astype(int).astype(str)

    # BMI：保持合理范围
    if "mean_bmi" in df.columns:
        bmi_values = pd.to_numeric(df["mean_bmi"], errors="coerce")
        bmi_valid_mask = bmi_values.notna()
        min_val, max_val = physiological_ranges["mean_bmi"]

        if bmi_valid_mask.any():
            valid_bmi = bmi_values[bmi_valid_mask]
            bmi_noise = np.random.normal(0, 0.2, bmi_valid_mask.sum())
            new_bmi = (valid_bmi + bmi_noise).clip(min_val, max_val)
            df.loc[bmi_valid_mask, "mean_bmi"] = new_bmi.round(1).astype(str)


# === 7. 性别比例微调 ===
def adjust_gender_ratio(df: pd.DataFrame) -> None:
    """微调性别比例，保持接近原始分布"""
    if "GENDER" not in df.columns:
        return

    current_male_ratio = (df["GENDER"] == "M").mean()
    target_male_ratio = 0.5116

    # 计算需要调整的数量
    n_total = len(df)
    n_target_male = int(n_total * target_male_ratio)
    n_current_male = (df["GENDER"] == "M").sum()

    n_change = abs(n_target_male - n_current_male)

    if n_change == 0:
        return

    if n_current_male > n_target_male:
        # 需要减少男性数量
        male_indices = df[df["GENDER"] == "M"].index
        if len(male_indices) > 0:
            change_indices = np.random.choice(male_indices, size=min(n_change, len(male_indices)), replace=False)
            df.loc[change_indices, "GENDER"] = "F"
    else:
        # 需要增加男性数量
        female_indices = df[df["GENDER"] == "F"].index
        if len(female_indices) > 0:
            change_indices = np.random.choice(female_indices, size=min(n_change, len(female_indices)), replace=False)
            df.loc[change_indices, "GENDER"] = "M"


# === 8. 列随机化（新增功能）===
def randomize_selected_columns(df: pd.DataFrame) -> None:
    """对选择的列进行随机化以增强匿名性"""

    # 选择对匿名性影响较小但对有用性影响不大的列
    columns_to_randomize = []

    # 医疗计数类列：轻微随机化
    medical_count_cols = ["num_immunizations", "num_allergies", "num_devices"]
    for col in medical_count_cols:
        if col in df.columns:
            columns_to_randomize.append(col)

    # 对选择的列进行轻微随机化
    for col in columns_to_randomize:
        if col in df.columns:
            # 获取当前列的数值
            current_values = pd.to_numeric(df[col], errors="coerce")
            valid_mask = current_values.notna()

            if valid_mask.any():
                valid_values = current_values[valid_mask]

                # 轻微随机化：90%的值保持不变，10%的值在±1范围内变化
                change_mask = (np.random.rand(valid_mask.sum()) < 0.1)
                if change_mask.any():
                    change_values = valid_values[change_mask]
                    # 在±1范围内随机变化
                    random_changes = np.random.choice([-1, 0, 1], size=change_mask.sum(), p=[0.4, 0.2, 0.4])
                    new_values = (change_values + random_changes).clip(0, None)  # 确保非负

                    # 更新数据
                    change_indices = valid_values[change_mask].index
                    df.loc[change_indices, col] = new_values.astype(int).astype(str)

                    print(f"列 {col} 随机化：改变了 {change_mask.sum()} 个值")


# === 9. 数据格式清理（修复年龄空值）===
def clean_data_format(df: pd.DataFrame) -> None:
    """确保数据符合CiDataFrame的格式要求"""

    # 首先处理年龄，确保没有空值
    if "AGE" in df.columns:
        age_num = pd.to_numeric(df["AGE"], errors="coerce")
        empty_age_mask = age_num.isna()
        if empty_age_mask.any():
            median_age = age_num.median()
            if pd.isna(median_age):
                median_age = 45
            df.loc[empty_age_mask, "AGE"] = str(int(median_age))
            print(f"在数据清理中修复了 {empty_age_mask.sum()} 个空年龄值")

    # 确保所有数值列都是有效的字符串格式且为整数
    numeric_cols = [
        "AGE", "encounter_count", "num_procedures", "num_medications",
        "num_immunizations", "num_allergies", "num_devices",
        "mean_systolic_bp", "mean_diastolic_bp", "mean_weight"
    ]

    # BMI可以保留小数
    float_cols = ["mean_bmi"]

    for col in numeric_cols:
        if col in df.columns:
            # 转换为数值，处理无效值
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            # 将有效数值转换为整数字符串，NaN用中位数填充
            median_val = numeric_series.median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = numeric_series.apply(
                lambda x: str(int(round(median_val))) if pd.isna(x) else str(int(round(x))))

    for col in float_cols:
        if col in df.columns:
            # 转换为数值，处理无效值
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            # 将有效数值转换为字符串，保留1位小数
            median_val = numeric_series.median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = numeric_series.apply(
                lambda x: f"{median_val:.1f}" if pd.isna(x) else f"{x:.1f}")

    # 确保分类变量是有效的字符串
    categorical_cols = ["GENDER", "RACE", "ETHNICITY"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": "", "None": "", "": ""})
            # 如果有空值，用最常见的值填充
            if df[col].isna().any() or (df[col] == "").any():
                most_common = df[col].mode()
                if len(most_common) > 0:
                    fill_value = most_common.iloc[0]
                    df[col] = df[col].replace({"": fill_value}).fillna(fill_value)

    # 确保标志变量是0/1或空字符串
    flag_cols = ["asthma_flag", "stroke_flag", "obesity_flag", "depression_flag"]
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": "", "None": "", "": ""})
            # 确保只有0和1，空值用0填充
            valid_mask = df[col].isin(["0", "1"])
            df.loc[~valid_mask, col] = "0"


def main():
    parser = argparse.ArgumentParser(description="增强版匿名化：在保持有用性70%+的前提下提高匿名性")
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

    # 应用增强匿名化方法
    print("开始应用增强匿名化...")

    # 1. 年龄组互换
    print("执行年龄组互换...")
    swap_age_groups(df)

    # 2. 种族合并为other
    print("执行种族合并为other...")
    merge_race_to_other(df)

    # 3. 民族安全扰动
    print("执行民族安全扰动...")
    safe_ethnicity_perturbation(df)

    # 4. 医疗计数微聚合
    print("执行医疗计数微聚合...")
    micro_aggregate_medical_counts(df)

    # 5. 疾病标志位条件扰动
    print("执行疾病标志位条件扰动...")
    conditional_flag_perturbation(df)

    # 6. 生理指标智能扰动
    print("执行生理指标智能扰动...")
    smart_physiological_perturbation(df)

    # 7. 性别比例微调
    print("执行性别比例微调...")
    adjust_gender_ratio(df)

    # 8. 列随机化（新增）
    print("执行列随机化...")
    randomize_selected_columns(df)

    # 9. 数据格式清理
    print("执行数据格式清理...")
    clean_data_format(df)

    # 输出结果
    Ci_df = CiDataFrame(df)
    Ci_df.to_csv(args.output_csv)
    print(f"✅ 增强匿名化完成！输出文件: {args.output_csv}")
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

    if "asthma_flag" in df.columns:
        asthma_rate = (df["asthma_flag"] == "1").mean()
        print(f"哮喘患病率: {asthma_rate * 100:.2f}%")


if __name__ == "__main__":
    main()