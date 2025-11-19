#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高有用性匿名化（v7 — preserve-LR）
要点：
1) 不改分类变量与所有 flag（尤其 asthma_flag）
2) 只对数值做 NaN 分层中位数填补；不做全局剪裁/取整（AGE 顶码例外）
3) 在以 asthma_flag 主导的分层内，对 [AGE + 计数 + 生理] 做“同一排列”的块置换
   —— 保留 (X|y) 的联合分布，显著降低 LR_asthma_diff；同时 stats_diff/KW 也更稳
"""

import sys, os
import argparse
import random
from typing import List, Dict

import numpy as np
import pandas as pd

# util
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'util'))
from pws_data_format import BiDataFrame, CiDataFrame

# ------------ 列配置 ------------
COUNT_COLS = [
    "encounter_count", "num_procedures", "num_medications",
    "num_immunizations", "num_allergies", "num_devices"
]
PHYSIO_COLS = [
    "mean_systolic_bp", "mean_diastolic_bp", "mean_weight", "mean_bmi"
]
FLAG_COLS = ["asthma_flag", "stroke_flag", "obesity_flag", "depression_flag"]
CAT_COLS  = ["GENDER", "ETHNICITY", "RACE"]

# 分层最小样本，回退时始终包含 asthma_flag
MIN_GROUP_SIZE = 9
SWAP_PROB = 1.0   # 满足规模就置换（可按需降到 0.8 增隐私随机性）

# ------------ 工具函数 ------------
def ensure_categoricals(df: pd.DataFrame):
    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = (df[col].astype(str)
                   .replace({"nan": "Unknown", "": "Unknown", "None": "Unknown"})
                   .fillna("Unknown"))
    return df

def ensure_flags(df: pd.DataFrame):
    for col in FLAG_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].where(df[col].isin(["0", "1"]), "0")
    return df

def topcode_age(df: pd.DataFrame):
    if "AGE" in df.columns:
        age = pd.to_numeric(df["AGE"], errors="coerce")
        # 缺失用全局中位数，顶码 90+
        med = np.nanmedian(age)
        if not np.isfinite(med):
            med = 45
        age = age.fillna(med)
        age.loc[age >= 90] = 90
        df["AGE"] = age
    return df

def build_perm_hierarchy(df: pd.DataFrame) -> List[List[str]]:
    """
    分层阶梯（始终保留 asthma_flag）：
    [GENDER,ETHNICITY,RACE,asthma] -> [GENDER,ETHNICITY,asthma]
    -> [GENDER,asthma] -> [asthma] -> []
    只保留 df 中存在的列，且去重
    """
    base = [
        ["GENDER", "ETHNICITY", "RACE", "asthma_flag"],
        ["GENDER", "ETHNICITY", "asthma_flag"],
        ["GENDER", "asthma_flag"],
        ["asthma_flag"],
        []
    ]
    out = []
    for g in base:
        cols = [c for c in g if c in df.columns]
        if not any(len(cols) == len(x) and all(a == b for a, b in zip(cols, x)) for x in out):
            out.append(cols)
    return out

def stratified_impute_nan(df: pd.DataFrame, col: str, hier: List[List[str]]):
    """
    仅对 NaN 做分层中位数填补；不裁剪、不改非缺失值。
    """
    x = pd.to_numeric(df[col], errors="coerce")
    bad = x.isna()
    if not bad.any():
        df[col] = x
        return

    for group_cols in hier:
        idx_bad = df.index[bad]
        if len(idx_bad) == 0:
            break

        if len(group_cols) > 0:
            gb = df.groupby(group_cols, dropna=False, observed=False)
            size_map = gb.size()
            med_map = gb.apply(lambda g: x.loc[g.index].median())
        else:
            size_global = len(df)
            med_global = x.median()

        fill_idx, fill_val = [], []
        for i in idx_bad:
            if len(group_cols) > 0:
                key = tuple(df.loc[i, gc] for gc in group_cols)
                med = med_map.get(key, np.nan) if size_map.get(key, 0) >= MIN_GROUP_SIZE else np.nan
            else:
                med = med_global if size_global >= MIN_GROUP_SIZE else np.nan
            if not pd.isna(med):
                fill_idx.append(i); fill_val.append(med)

        if fill_idx:
            x.loc[fill_idx] = fill_val
            bad.loc[fill_idx] = False

    # 兜底：全局中位数
    if bad.any():
        med = x.median()
        if pd.isna(med):
            med = 0.0
        x.loc[bad] = med

    df[col] = x

def block_permute_bundle(df: pd.DataFrame, bundle_cols: List[str], hier: List[List[str]], rng: np.random.Generator):
    """
    在每个层级内：对 bundle 列使用“同一随机排列”进行置换。
    —— 注意：bundle = [AGE] + 计数 + 生理（AGE 也一起 permute！）
    —— 这样 X|asthma_flag 的联合分布保持（行重排），LR 系数基本不变。
    """
    placed = pd.Series(False, index=df.index)

    for group_cols in hier:
        if len(group_cols) > 0:
            gb = df[~placed].groupby(group_cols, dropna=False, observed=False)
            iterator = gb
        else:
            key = ("__GLOBAL__",)
            gb = {key: df[~placed]}
            iterator = gb.items()

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

# ------------ 主流程 ------------
def main():
    ap = argparse.ArgumentParser(description="高有用性匿名化 v7（preserve-LR）")
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    df = BiDataFrame.read_csv(args.input_csv)
    print(f"原始数据行数: {len(df)}")

    # 分类/标志清理（不改变取值，仅规范空值）
    df = ensure_categoricals(df)
    df = ensure_flags(df)

    # AGE 顶码（安全但不影响 LR）：90+ -> 90；缺失用全局中位数
    df = topcode_age(df)

    # 数值列集合：把 AGE 放进 bundle，一起 permute
    num_cols = []
    if "AGE" in df.columns: num_cols.append("AGE")
    num_cols += [c for c in COUNT_COLS if c in df.columns]
    num_cols += [c for c in PHYSIO_COLS if c in df.columns]

    # 仅对 NaN 做分层中位数填补（按以 asthma 为核心的分层）
    hier = build_perm_hierarchy(df)
    for col in num_cols:
        stratified_impute_nan(df, col, hier)

    # 分层“同排列”置换（保持 X|asthma 联合分布）
    bundle_cols = list(num_cols)  # 复制
    if len(bundle_cols) >= 2:
        block_permute_bundle(df, bundle_cols, hier, rng)

    # —— 输出格式：尽量不改变数值精度，直接写出 —— #
    # 注意：CiDataFrame 内部会自行校验；这里不做额外取整/四舍五入
    Ci_df = CiDataFrame(df)
    Ci_df.to_csv(args.output_csv)
    print(f"✅ v7 完成，已写出: {args.output_csv}")
    print(f"处理后数据行数: {len(df)}")

if __name__ == "__main__":
    main()
