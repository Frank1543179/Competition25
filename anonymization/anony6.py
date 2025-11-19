#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高有用性匿名化（v6.1）
策略：分层中位数填补 + 分层“块置换”（同组同排列），不改分类/标志位，
在尽量保留与 asthma_flag 的相关结构的同时打散可链接性，优化
stats_diff / LR_asthma_diff / KW_IND_diff 综合得分。
"""

import sys, os
import argparse
import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# util
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'util'))
from pws_data_format import BiDataFrame, CiDataFrame

# -------------------------
# 配置
# -------------------------
COUNT_COLS = [
    "encounter_count", "num_procedures", "num_medications",
    "num_immunizations", "num_allergies", "num_devices"
]

PHYSIO_COLS = [
    "mean_systolic_bp", "mean_diastolic_bp", "mean_weight", "mean_bmi"
]

# 合法范围（仅用于裁剪与缺失修补）
RANGES: Dict[str, Tuple[float, float]] = {
    "AGE": (2, 110),
    "encounter_count": (0, 2000),
    "num_procedures": (0, 500),
    "num_medications": (0, 1000),
    "num_immunizations": (0, 300),
    "num_allergies": (0, 200),
    "num_devices": (0, 100),
    "mean_systolic_bp": (60, 220),     # mmHg
    "mean_diastolic_bp": (35, 130),    # mmHg
    "mean_weight": (25, 300),          # kg
    "mean_bmi": (12.0, 65.0),
}

MIN_GROUP_SIZE = 9       # 小组回退阈值
SWAP_PROB = 1.0          # 组满足规模时执行置换的概率

# -------------------------
# 工具函数
# -------------------------
def build_hier_groups(df: pd.DataFrame) -> List[List[str]]:
    """
    动态构建层级分组：只保留当前 df 中存在的列，并去重。
    排序从细到粗（首层尽量包含 asthma_flag）。
    """
    base = [
        ["GENDER", "ETHNICITY", "RACE", "AGE_BIN", "asthma_flag"],
        ["GENDER", "ETHNICITY", "RACE", "AGE_BIN"],
        ["GENDER", "AGE_BIN"],
        ["AGE_BIN"],
        []
    ]
    out: List[List[str]] = []
    for grp in base:
        g = [c for c in grp if c in df.columns]
        # 去重（把相同的层级只保留一次）
        if not any(len(g) == len(x) and all(a == b for a, b in zip(g, x)) for x in out):
            out.append(g)
    return out

def _age_bins(series: pd.Series) -> pd.Series:
    age = pd.to_numeric(series, errors="coerce")
    med = np.nanmedian(age)
    if not np.isfinite(med):
        med = 45
    age = age.fillna(med).clip(*RANGES["AGE"])
    bins = [0, 18, 30, 40, 50, 65, 80, 120]
    labels = ['<=18', '19-30', '31-40', '41-50', '51-65', '66-80', '80+']
    return pd.cut(age, bins=bins, labels=labels, include_lowest=True, right=True)

def clip_to_range(s: pd.Series, name: str) -> pd.Series:
    lo, hi = RANGES.get(name, (-np.inf, np.inf))
    return pd.to_numeric(s, errors="coerce").clip(lo, hi)

def ensure_categoricals(df: pd.DataFrame):
    for col in ["GENDER", "ETHNICITY", "RACE"]:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].astype(str).replace({"nan": "Unknown", "": "Unknown"}).fillna("Unknown")
    return df

def ensure_flags(df: pd.DataFrame):
    for col in ["asthma_flag", "stroke_flag", "obesity_flag", "depression_flag"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].where(df[col].isin(["0", "1"]), "0")
    return df

def stratified_impute_numeric(df: pd.DataFrame, col: str, hier_groups: List[List[str]]):
    """
    分层中位数填补：按 hier_groups 逐级回退，填补缺失或越界值；最后裁剪到范围。
    用事先转好的数值 x 计算中位数，避免 dtype 问题。
    """
    x = pd.to_numeric(df[col], errors='coerce')
    lo, hi = RANGES.get(col, (-np.inf, np.inf))
    bad = x.isna() | (x < lo) | (x > hi)

    if not bad.any():
        df[col] = x.clip(lo, hi)
        return

    for group_cols in hier_groups:
        idx_bad = df.index[bad]
        if len(idx_bad) == 0:
            break

        if len(group_cols) > 0:
            gb = df.groupby(group_cols, dropna=False, observed=False)
            size_map = gb.size()
            # 关键：用 x（数值化后的列）按组取中位数，避免 SeriesGroupBy 的 numeric_only 限制
            med_map = gb.apply(lambda g: x.loc[g.index].median())
        else:
            med_global = x.median()
            size_global = len(df)

        to_fill_idx = []
        to_fill_val = []
        for i in idx_bad:
            if len(group_cols) > 0:
                key = tuple(df.loc[i, gc] for gc in group_cols)
                if size_map.get(key, 0) >= MIN_GROUP_SIZE:
                    med = med_map.get(key, np.nan)
                else:
                    med = np.nan
            else:
                med = med_global if size_global >= MIN_GROUP_SIZE else np.nan

            if not pd.isna(med):
                to_fill_idx.append(i)
                to_fill_val.append(med)

        if to_fill_idx:
            x.loc[to_fill_idx] = to_fill_val
            bad.loc[to_fill_idx] = False

    # 兜底：全局中位数或下界
    if bad.any():
        med = x.median()
        if pd.isna(med):
            med = lo
        x.loc[bad] = med

    df[col] = x.clip(lo, hi)

def block_permute_within_groups(
    df: pd.DataFrame,
    bundle_cols: List[str],
    hier_groups: List[List[str]],
    rng: np.random.Generator
):
    """
    在多级分层中对 bundle 列做 '块置换'：
    - 仅在组规模 >= MIN_GROUP_SIZE 且随机命中 SWAP_PROB 时执行
    - 同一组对 bundle_cols 使用同一随机排列，保留簇内相关结构
    - 先在最细分层（含 asthma_flag）尝试，规模不足再回退
    """
    placed = pd.Series(False, index=df.index)

    for group_cols in hier_groups:
        if len(group_cols) > 0:
            gb = df[~placed].groupby(group_cols, dropna=False, observed=False)
            iterator = gb  # (key, group)
        else:
            key = ("__GLOBAL__",)
            gb = {key: df[~placed]}
            iterator = gb.items()  # dict 需 .items()

        for key, g in iterator:
            idx = g.index
            n = len(idx)
            if n < MIN_GROUP_SIZE:
                continue
            if rng.random() >= SWAP_PROB:
                continue

            perm = rng.permutation(n)
            for col in bundle_cols:
                if col not in df.columns:
                    continue
                vals = df.loc[idx, col].to_numpy()
                df.loc[idx, col] = vals[perm]

            placed.loc[idx] = True

# -------------------------
# 主流程
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="高有用性匿名化（v6.1）：分层中位数填补 + 分层块置换（不动分类与标志位）")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 随机种子
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 读取 Bi
    df = BiDataFrame.read_csv(args.input_csv)
    print(f"原始数据行数: {len(df)}")

    # —— 基础清理 —— #
    df = ensure_categoricals(df)
    df = ensure_flags(df)

    # AGE：裁剪 + 轻度顶码（≥90 记 90）以降重标识风险；尽量不改整体分布
    if "AGE" in df.columns:
        age = clip_to_range(df["AGE"], "AGE")
        age = age.clip(*RANGES["AGE"])
        age.loc[age >= 90] = 90
        df["AGE"] = age
        df["AGE_BIN"] = _age_bins(df["AGE"])
    else:
        df["AGE_BIN"] = pd.Series(["41-50"] * len(df), index=df.index)

    # 动态层级（只包含存在的列）
    hier_groups = build_hier_groups(df)

    # 数值列集合（不包含 AGE，AGE 仅用于分层，不置换）
    num_cols = [c for c in COUNT_COLS + PHYSIO_COLS if c in df.columns]

    # —— 分层中位数填补（缺失/越界）+ 裁剪 —— #
    for col in (["AGE"] + num_cols):
        if col in df.columns:
            stratified_impute_numeric(df, col, hier_groups)

    # —— 分层块置换（bundle 一起 permute） —— #
    bundle_cols = [c for c in num_cols]
    if len(bundle_cols) >= 2:
        block_permute_within_groups(df, bundle_cols, hier_groups, rng)

    # —— 输出前格式规范化 —— #
    # AGE -> int -> str
    if "AGE" in df.columns:
        df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce").round().astype(int).astype(str)

    # 计数列 -> int -> str
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round().astype(int).astype(str)

    # 生理列
    if "mean_systolic_bp" in df.columns:
        df["mean_systolic_bp"] = pd.to_numeric(df["mean_systolic_bp"], errors="coerce").round().astype(int).astype(str)
    if "mean_diastolic_bp" in df.columns:
        df["mean_diastolic_bp"] = pd.to_numeric(df["mean_diastolic_bp"], errors="coerce").round().astype(int).astype(str)
    if "mean_weight" in df.columns:
        df["mean_weight"] = pd.to_numeric(df["mean_weight"], errors="coerce").round().astype(int).astype(str)
    if "mean_bmi" in df.columns:
        df["mean_bmi"] = pd.to_numeric(df["mean_bmi"], errors="coerce").round(1).astype(str)

    # 分类&标志位确保字符串合法
    for col in ["GENDER", "ETHNICITY", "RACE"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": "Unknown", "": "Unknown"}).fillna("Unknown")

    for col in ["asthma_flag", "stroke_flag", "obesity_flag", "depression_flag"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].where(df[col].isin(["0", "1"]), "0")

    # 删除中间列
    if "AGE_BIN" in df.columns:
        df.drop(columns=["AGE_BIN"], inplace=True)

    # 写出 Ci
    Ci_df = CiDataFrame(df)
    Ci_df.to_csv(args.output_csv)
    print(f"✅ 高有用性匿名化（v6.1）完成！输出文件: {args.output_csv}")
    print(f"处理后数据行数: {len(df)}")


if __name__ == "__main__":
    main()
