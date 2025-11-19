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


# === 1. 扩展年龄组互换策略 ===
def swap_age_groups(df: pd.DataFrame) -> None:
    """对更多年龄组进行互换，增加扰动范围"""
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
        print(f"修复了 {empty_age_mask.sum()} 个空年龄值")

    age_num = pd.to_numeric(df["AGE"], errors="coerce")

    # 扩展年龄组互换：增加更多组合
    age_swap_groups = [
        # (年轻组范围, 年长组范围, 互换比例)
        ((21, 30), (31, 40), 0.12),  # 提高比例
        ((31, 40), (41, 50), 0.10),
        ((51, 60), (61, 70), 0.12),  # 提高比例
        ((41, 50), (61, 70), 0.08),  # 新增跨代互换
        ((18, 25), (66, 75), 0.05)  # 新增极端互换
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

            print(f"年龄组 {young_min}-{young_max} ↔ {old_min}-{old_max}: 互换 {n_swap} 条记录")


# === 2. 扩展种族合并策略 ===
def merge_race_to_other(df: pd.DataFrame) -> None:
    """将更多种族类别合并为other，增加匿名性"""
    if "RACE" not in df.columns:
        return

    # 扩展要合并的种族类别
    races_to_merge = ['hawaiian', 'native', 'asian', 'other']  # 增加asian和other

    # 计算原始分布
    original_counts = df["RACE"].value_counts()
    print("原始种族分布:")
    for race, count in original_counts.items():
        print(f"  {race}: {count}人 ({count / len(df) * 100:.2f}%)")

    # 将这些种族的记录合并为"other"，但保留部分asian
    mask = df["RACE"].isin(['hawaiian', 'native', 'other'])
    asian_mask = (df["RACE"] == "asian") & (np.random.rand(len(df)) < 0.3)  # 30%的asian改为other

    df.loc[mask, "RACE"] = "other"
    df.loc[asian_mask, "RACE"] = "other"

    print(f"种族合并完成：将 {mask.sum() + asian_mask.sum()} 条记录的种族合并为 'other'")

    # 显示合并后分布
    new_counts = df["RACE"].value_counts()
    print("合并后种族分布:")
    for race, count in new_counts.items():
        print(f"  {race}: {count}人 ({count / len(df) * 100:.2f}%)")


# === 3. 增强民族扰动 ===
def safe_ethnicity_perturbation(df: pd.DataFrame) -> None:
    """增加民族扰动的强度和范围"""
    if "ETHNICITY" not in df.columns:
        return

    # 记录原始分布
    original_dist = df["ETHNICITY"].value_counts(normalize=True)
    print("原始民族分布:")
    for eth, prop in original_dist.items():
        print(f"  {eth}: {prop * 100:.2f}%")

    # 增加扰动比例和范围
    nonhispanic_mask = (df["ETHNICITY"] == "nonhispanic") & (np.random.rand(len(df)) < 0.08)  # 提高到8%
    hispanic_mask = (df["ETHNICITY"] == "hispanic") & (np.random.rand(len(df)) < 0.06)  # 提高到6%
    unknown_mask = (df["ETHNICITY"] == "unknown") & (np.random.rand(len(df)) < 0.15)  # unknown扰动

    # 非西班牙裔改为西班牙裔
    df.loc[nonhispanic_mask, "ETHNICITY"] = "hispanic"
    # 西班牙裔改为非西班牙裔
    df.loc[hispanic_mask, "ETHNICITY"] = "nonhispanic"
    # unknown随机分配
    df.loc[unknown_mask, "ETHNICITY"] = np.random.choice(["hispanic", "nonhispanic"], unknown_mask.sum())

    # 显示扰动后分布
    new_dist = df["ETHNICITY"].value_counts(normalize=True)
    print("扰动后民族分布:")
    for eth, prop in new_dist.items():
        print(f"  {eth}: {prop * 100:.2f}%")


# === 4. 增强医疗计数指标的微聚合 ===
def micro_aggregate_medical_counts(df: pd.DataFrame) -> None:
    """增加医疗计数指标的扰动强度"""
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

        # 记录原始统计
        original_values = pd.to_numeric(df[col], errors="coerce")
        original_mean = original_values.mean()
        original_std = original_values.std()

        # 使用更细的年龄分组
        age_bins = [0, 18, 30, 45, 60, 75, 200]
        age_labels = ['0-17', '18-29', '30-44', '45-59', '60-74', '75+']

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
                std_val = max(1, mean_val * 0.15)  # 增加基础标准差

            # 增加噪声强度
            noise_std = std_val * 0.08  # 提高到8%
            noise = np.random.normal(0, noise_std, size=valid_mask.sum())

            # 对部分记录进行更大范围的扰动
            large_noise_mask = np.random.rand(valid_mask.sum()) < 0.1  # 10%的记录使用更大噪声
            if large_noise_mask.any():
                large_noise = np.random.normal(0, std_val * 0.2, size=large_noise_mask.sum())
                noise[large_noise_mask] = large_noise

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

        # 显示统计变化
        new_values = pd.to_numeric(df[col], errors="coerce")
        new_mean = new_values.mean()
        print(
            f"{col}: 均值 {original_mean:.2f} → {new_mean:.2f}, 变化 {((new_mean - original_mean) / original_mean * 100):+.1f}%")


# === 5. 增强疾病标志位的条件扰动 ===
def conditional_flag_perturbation(df: pd.DataFrame) -> None:
    """显著增加疾病标志位的扰动，特别是asthma_flag"""
    if "AGE" not in df.columns:
        return

    age_num = pd.to_numeric(df["AGE"], errors="coerce")

    # asthma_flag: 显著增加扰动以影响LR_asthma_diff
    if "asthma_flag" in df.columns:
        original_asthma_rate = (df["asthma_flag"] == "1").mean()
        print(f"原始哮喘患病率: {original_asthma_rate * 100:.2f}%")

        # 策略1: 基于年龄的定向扰动
        # 儿童和年轻人更容易患哮喘
        young_asthma_mask = (age_num < 18) & (df["asthma_flag"] == "0")
        adult_asthma_mask = (age_num >= 18) & (age_num < 40) & (df["asthma_flag"] == "0")
        elderly_no_asthma_mask = (age_num > 60) & (df["asthma_flag"] == "1")

        # 增加儿童哮喘病例 (15%)
        if young_asthma_mask.any():
            young_add = df[young_asthma_mask].sample(frac=0.15, random_state=42).index
            df.loc[young_add, "asthma_flag"] = "1"
            print(f"增加 {len(young_add)} 个儿童哮喘病例")

        # 增加年轻成人哮喘病例 (8%)
        if adult_asthma_mask.any():
            adult_add = df[adult_asthma_mask].sample(frac=0.08, random_state=42).index
            df.loc[adult_add, "asthma_flag"] = "1"
            print(f"增加 {len(adult_add)} 个年轻成人哮喘病例")

        # 减少老年人哮喘病例 (12%)
        if elderly_no_asthma_mask.any():
            elderly_remove = df[elderly_no_asthma_mask].sample(frac=0.12, random_state=42).index
            df.loc[elderly_remove, "asthma_flag"] = "0"
            print(f"减少 {len(elderly_remove)} 个老年人哮喘病例")

        # 策略2: 全局随机扰动 (5%)
        global_flip_mask = (df["asthma_flag"].isin(["0", "1"])) & (np.random.rand(len(df)) < 0.05)
        df.loc[global_flip_mask & (df["asthma_flag"] == "0"), "asthma_flag"] = "1"
        df.loc[global_flip_mask & (df["asthma_flag"] == "1"), "asthma_flag"] = "0"
        print(f"全局随机翻转 {global_flip_mask.sum()} 个哮喘标志位")

        new_asthma_rate = (df["asthma_flag"] == "1").mean()
        print(f"扰动后哮喘患病率: {new_asthma_rate * 100:.2f}%")
        print(f"哮喘患病率变化: {(new_asthma_rate - original_asthma_rate) * 100:+.2f}%")

    # stroke_flag: 增强扰动
    if "stroke_flag" in df.columns:
        original_stroke_rate = (df["stroke_flag"] == "1").mean()

        # 老年人中风标志位扰动
        elderly_stroke_mask = (age_num > 65) & (df["stroke_flag"] == "1")
        middle_no_stroke_mask = (age_num >= 45) & (age_num <= 65) & (df["stroke_flag"] == "0")

        if elderly_stroke_mask.any():
            # 8%的老年人中风改为无中风
            elderly_flip = df[elderly_stroke_mask].sample(frac=0.08, random_state=42).index
            df.loc[elderly_flip, "stroke_flag"] = "0"

        if middle_no_stroke_mask.any():
            # 3%的中年人无中风改为有中风
            middle_flip = df[middle_no_stroke_mask].sample(frac=0.03, random_state=42).index
            df.loc[middle_flip, "stroke_flag"] = "1"

    # obesity_flag: 增强扰动
    if "obesity_flag" in df.columns:
        middle_aged_obese = (age_num >= 40) & (age_num <= 60) & (df["obesity_flag"] == "0")
        young_obese = (age_num < 30) & (df["obesity_flag"] == "1")

        if middle_aged_obese.any():
            # 10%的中年无肥胖改为有肥胖
            middle_flip_indices = df[middle_aged_obese].sample(frac=0.10, random_state=42).index
            df.loc[middle_flip_indices, "obesity_flag"] = "1"

        if young_obese.any():
            # 12%的年轻有肥胖改为无肥胖
            young_obese_flip = df[young_obese].sample(frac=0.12, random_state=42).index
            df.loc[young_obese_flip, "obesity_flag"] = "0"

    # depression_flag: 调整患病率
    if "depression_flag" in df.columns:
        current_depression_rate = (df["depression_flag"] == "1").mean()
        target_depression_rate = 0.045  # 提高到4.5%

        if current_depression_rate < target_depression_rate:
            n_total = len(df)
            n_target_depressed = int(n_total * target_depression_rate)
            n_current_depressed = (df["depression_flag"] == "1").sum()
            n_to_add = n_target_depressed - n_current_depressed

            if n_to_add > 0:
                # 优先选择年轻人和女性
                young_adult_mask = (age_num >= 18) & (age_num <= 40) & (df["depression_flag"] == "0")
                female_mask = (df["GENDER"] == "F") & (df["depression_flag"] == "0")

                priority_mask = young_adult_mask & female_mask
                secondary_mask = young_adult_mask | female_mask

                priority_indices = df[priority_mask].index.tolist()
                if len(priority_indices) >= n_to_add:
                    add_indices = random.sample(priority_indices, n_to_add)
                else:
                    add_indices = priority_indices
                    remaining = n_to_add - len(priority_indices)
                    secondary_candidates = list(set(df[secondary_mask].index.tolist()) - set(priority_indices))
                    if len(secondary_candidates) >= remaining:
                        add_indices.extend(random.sample(secondary_candidates, remaining))
                    else:
                        all_non_depressed = df[df["depression_flag"] == "0"].index.tolist()
                        remaining_candidates = list(set(all_non_depressed) - set(add_indices))
                        if len(remaining_candidates) >= remaining:
                            add_indices.extend(random.sample(remaining_candidates, remaining))

                df.loc[add_indices, "depression_flag"] = "1"
                print(
                    f"增加了 {len(add_indices)} 个抑郁症病例，患病率 {current_depression_rate * 100:.2f}% → {target_depression_rate * 100:.2f}%")


# === 6. 增强生理指标的智能扰动 ===
def smart_physiological_perturbation(df: pd.DataFrame) -> None:
    """增加生理指标的扰动强度"""

    physiological_ranges = {
        "mean_systolic_bp": (60, 200),
        "mean_diastolic_bp": (40, 120),
        "mean_weight": (30, 200),
        "mean_bmi": (15, 50)
    }

    # 血压指标：增强扰动
    for bp_col in ["mean_systolic_bp", "mean_diastolic_bp"]:
        if bp_col in df.columns:
            bp_values = pd.to_numeric(df[bp_col], errors="coerce")
            valid_mask = bp_values.notna()
            min_val, max_val = physiological_ranges[bp_col]

            if valid_mask.any():
                valid_bp = bp_values[valid_mask]

                # 增加噪声强度
                if bp_col == "mean_systolic_bp":
                    normal_noise_std = 2.5  # 提高噪声
                    adjustment_std = 3.0
                else:
                    normal_noise_std = 1.8
                    adjustment_std = 2.2

                # 正常范围扰动
                if bp_col == "mean_systolic_bp":
                    normal_bp_mask = (valid_bp >= 90) & (valid_bp <= 140)
                else:
                    normal_bp_mask = (valid_bp >= 60) & (valid_bp <= 90)

                abnormal_bp_mask = ~normal_bp_mask

                normal_noise = np.random.normal(0, normal_noise_std, normal_bp_mask.sum())
                valid_bp.loc[normal_bp_mask] = (valid_bp[normal_bp_mask] + normal_noise).clip(
                    physiological_ranges[bp_col][0], physiological_ranges[bp_col][1])

                # 异常值调整
                if abnormal_bp_mask.any():
                    adjustment = np.random.normal(4, adjustment_std, abnormal_bp_mask.sum())
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

    # 体重和BMI：增强扰动
    for col in ["mean_weight", "mean_bmi"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            valid_mask = values.notna()
            min_val, max_val = physiological_ranges[col]

            if valid_mask.any():
                valid_values = values[valid_mask]

                if col == "mean_weight":
                    noise_std = 1.2  # 提高噪声
                else:
                    noise_std = 0.4

                noise = np.random.normal(0, noise_std, valid_mask.sum())
                new_values = (valid_values + noise).clip(min_val, max_val)

                if col == "mean_weight":
                    df.loc[valid_mask, col] = new_values.round(0).astype(int).astype(str)
                else:
                    df.loc[valid_mask, col] = new_values.round(1).astype(str)


# === 7. 增强性别比例微调 ===
def adjust_gender_ratio(df: pd.DataFrame) -> None:
    """增加性别扰动的强度"""
    if "GENDER" not in df.columns:
        return

    original_male_ratio = (df["GENDER"] == "M").mean()
    target_male_ratio = 0.5116

    n_total = len(df)
    n_target_male = int(n_total * target_male_ratio)
    n_current_male = (df["GENDER"] == "M").sum()

    n_change = abs(n_target_male - n_current_male)

    # 增加额外扰动：在达到目标比例后额外扰动2%的记录
    extra_perturbation = int(n_total * 0.02)
    n_change_total = n_change + extra_perturbation

    if n_current_male > n_target_male:
        male_indices = df[df["GENDER"] == "M"].index
        if len(male_indices) > 0:
            change_indices = np.random.choice(male_indices, size=min(n_change_total, len(male_indices)), replace=False)
            df.loc[change_indices, "GENDER"] = "F"
    else:
        female_indices = df[df["GENDER"] == "F"].index
        if len(female_indices) > 0:
            change_indices = np.random.choice(female_indices, size=min(n_change_total, len(female_indices)),
                                              replace=False)
            df.loc[change_indices, "GENDER"] = "M"

    new_male_ratio = (df["GENDER"] == "M").mean()
    print(f"性别比例调整: {original_male_ratio * 100:.2f}% → {new_male_ratio * 100:.2f}%")


# === 8. 增强列随机化 ===
def randomize_selected_columns(df: pd.DataFrame) -> None:
    """显著增强列随机化强度"""

    # 扩展随机化列的范围
    columns_to_randomize = [
        "num_immunizations", "num_allergies", "num_devices",
        "num_procedures", "num_medications"
    ]

    total_changes = 0

    for col in columns_to_randomize:
        if col in df.columns:
            current_values = pd.to_numeric(df[col], errors="coerce")
            valid_mask = current_values.notna()

            if valid_mask.any():
                valid_values = current_values[valid_mask]

                # 提高随机化比例到20%
                change_mask = (np.random.rand(valid_mask.sum()) < 0.20)

                if change_mask.any():
                    change_values = valid_values[change_mask]

                    # 根据列类型确定扰动范围
                    if col in ["num_immunizations", "num_allergies", "num_devices"]:
                        # 小数值列：±2范围内变化
                        random_changes = np.random.choice([-2, -1, 0, 1, 2],
                                                          size=change_mask.sum(),
                                                          p=[0.15, 0.2, 0.3, 0.2, 0.15])
                    else:
                        # 大数值列：±5%范围内变化，但至少±1
                        changes = (change_values * 0.05 * np.random.choice([-1, 1], size=change_mask.sum())).round(0)
                        changes = np.where(np.abs(changes) < 1, np.sign(changes), changes)
                        random_changes = changes.astype(int)

                    new_values = (change_values + random_changes).clip(0, None)

                    change_indices = valid_values[change_mask].index
                    df.loc[change_indices, col] = new_values.astype(int).astype(str)

                    total_changes += change_mask.sum()
                    print(f"列 {col} 随机化：改变了 {change_mask.sum()} 个值")

    print(f"总计随机化改变: {total_changes} 个值")


# === 9. 新增记录交换功能 ===
def swap_complete_records(df: pd.DataFrame) -> None:
    """交换完整记录以增强匿名性"""
    n_swaps = int(len(df) * 0.03)  # 交换3%的记录

    if n_swaps > 1:
        # 选择要交换的记录对
        indices = df.index.tolist()
        swap_pairs = []

        for _ in range(n_swaps):
            if len(indices) >= 2:
                pair = random.sample(indices, 2)
                swap_pairs.append(pair)
                # 从候选池中移除已选择的索引
                indices = [i for i in indices if i not in pair]

        # 执行记录交换
        for i, j in swap_pairs:
            # 交换除ID外的所有列
            cols_to_swap = [col for col in df.columns if col not in ['ID', 'id']]
            temp = df.loc[i, cols_to_swap].copy()
            df.loc[i, cols_to_swap] = df.loc[j, cols_to_swap]
            df.loc[j, cols_to_swap] = temp

        print(f"完成了 {len(swap_pairs)} 对记录交换")


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
    parser = argparse.ArgumentParser(description="增强版匿名化：在保持有用性70%+的前提下提高匿名性")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（再現用）")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    df = BiDataFrame.read_csv(args.input_csv)
    print(f"原始数据行数: {len(df)}")
    print("开始应用增强匿名化...")

    # 执行所有匿名化步骤
    steps = [
        ("年龄组互换", swap_age_groups),
        ("种族合并", merge_race_to_other),
        ("民族扰动", safe_ethnicity_perturbation),
        ("医疗计数微聚合", micro_aggregate_medical_counts),
        ("疾病标志位扰动", conditional_flag_perturbation),
        ("生理指标扰动", smart_physiological_perturbation),
        ("性别比例调整", adjust_gender_ratio),
        ("列随机化", randomize_selected_columns),
        ("记录交换", swap_complete_records),
        ("数据格式清理", clean_data_format)
    ]

    for step_name, step_func in steps:
        print(f"\n=== 执行 {step_name} ===")
        step_func(df)

    Ci_df = CiDataFrame(df)
    Ci_df.to_csv(args.output_csv)
    print(f"\n✅ 增强匿名化完成！输出文件: {args.output_csv}")
    print(f"处理后数据行数: {len(df)}")

    # 显示最终统计信息
    if "RACE" in df.columns:
        race_counts = df["RACE"].value_counts()
        print("\n最终种族分布:")
        for race, count in race_counts.items():
            print(f"  {race}: {count}人 ({count / len(df) * 100:.2f}%)")

    if "depression_flag" in df.columns:
        depression_rate = (df["depression_flag"] == "1").mean()
        print(f"最终抑郁症患病率: {depression_rate * 100:.2f}%")

    if "asthma_flag" in df.columns:
        asthma_rate = (df["asthma_flag"] == "1").mean()
        print(f"最终哮喘患病率: {asthma_rate * 100:.2f}%")

    if "GENDER" in df.columns:
        male_rate = (df["GENDER"] == "M").mean()
        print(f"最终男性比例: {male_rate * 100:.2f}%")


if __name__ == "__main__":
    main()