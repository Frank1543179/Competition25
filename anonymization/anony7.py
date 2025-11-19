#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 保相关增强版匿名化：最小改动、显著压低 LR_asthma_diff、保 stats/KW 稳定
import sys, os
import argparse
import random
from typing import List

import numpy as np
import pandas as pd

# 相对导入 util
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'util'))
from pws_data_format import BiDataFrame, CiDataFrame

# ===== 列配置 =====
COUNT_COLS = [
    "encounter_count", "num_procedures", "num_medications",
    "num_immunizations", "num_allergies", "num_devices"
]
PHYSIO_COLS = ["mean_systolic_bp", "mean_diastolic_bp", "mean_weight", "mean_bmi"]
FLAG_COLS   = ["asthma_flag", "stroke_flag", "obesity_flag", "depression_flag"]
CAT_COLS    = ["GENDER", "RACE", "ETHNICITY"]

MIN_GROUP_SIZE = 7
SWAP_PROB = 0.8  # 分层规模达标即置换（如需更随机可降到 0.8）

# ===== 工具函数 =====
def ensure_categoricals(df: pd.DataFrame) -> None:
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "Unknown"
        df[c] = (df[c].astype(str)
                       .replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
                       ).fillna("Unknown")

def ensure_flags(df: pd.DataFrame) -> None:
    for c in FLAG_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)
            df[c] = df[c].where(df[c].isin(["0","1"]), "0")

def _age_bins(series: pd.Series) -> pd.Series:
    age = pd.to_numeric(series, errors="coerce")
    med = np.nanmedian(age)
    if not np.isfinite(med): med = 45
    age = age.fillna(med)
    bins = [0, 18, 30, 40, 50, 65, 80, 120]
    labels = ['<=18', '19-30', '31-40', '41-50', '51-65', '66-80', '80+']
    return pd.cut(age, bins=bins, labels=labels, include_lowest=True, right=True)

def build_hierarchy(df: pd.DataFrame) -> List[List[str]]:
    # 以 asthma_flag 为核心，逐步回退；仅使用现有列
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
        if not any(len(cols)==len(x) and all(a==b for a,b in zip(cols,x)) for x in out):
            out.append(cols)
    return out

def stratified_impute_nan(df: pd.DataFrame, col: str, hier: List[List[str]]) -> None:
    """仅补 NaN：按分层取组内中位数，组小则回退；不触碰非缺失值。"""
    x = pd.to_numeric(df[col], errors="coerce")
    bad = x.isna()
    if not bad.any():
        df[col] = x
        return
    for group_cols in hier:
        idx_bad = df.index[bad]
        if len(idx_bad)==0: break
        if len(group_cols)>0:
            gb = df.groupby(group_cols, dropna=False, observed=False)
            size_map = gb.size()
            med_map  = gb.apply(lambda g: x.loc[g.index].median())
        else:
            size_global = len(df)
            med_global  = x.median()
        fill_idx, fill_val = [], []
        for i in idx_bad:
            if len(group_cols)>0:
                key = tuple(df.loc[i, gc] for gc in group_cols)
                med = med_map.get(key, np.nan) if size_map.get(key,0)>=MIN_GROUP_SIZE else np.nan
            else:
                med = med_global if size_global>=MIN_GROUP_SIZE else np.nan
            if not pd.isna(med):
                fill_idx.append(i); fill_val.append(med)
        if fill_idx:
            x.loc[fill_idx] = fill_val
            bad.loc[fill_idx] = False
    if bad.any():
        med = x.median()
        if pd.isna(med): med = 0.0
        x.loc[bad] = med
    df[col] = x

def block_permute_bundle(df: pd.DataFrame, bundle_cols: List[str], hier: List[List[str]], rng: np.random.Generator) -> None:
    """在每个分层里对 bundle 列应用“同一随机排列”的块置换（保持 X|asthma 联合分布）。"""
    placed = pd.Series(False, index=df.index)
    for group_cols in hier:
        if len(group_cols)>0:
            gb = df[~placed].groupby(group_cols, dropna=False, observed=False)
            iterator = gb  # yields (key, group)
        else:
            iterator = {( "__GLOBAL__", ): df[~placed]}.items()
        for key, g in iterator:
            idx = g.index
            n = len(idx)
            if n < MIN_GROUP_SIZE: continue
            if rng.random() >= SWAP_PROB: continue
            perm = rng.permutation(n)
            for c in bundle_cols:
                if c not in df.columns: continue
                vals = df.loc[idx, c].to_numpy()
                df.loc[idx, c] = vals[perm]
            placed.loc[idx] = True

def tiny_randomize_one_count(df: pd.DataFrame, col: str, rng: np.random.Generator,
                             frac: float = 0.003, delta_choices = (-1, +1),
                             lo: int = 0, hi: int = 100):
    """
    对单列做极小幅随机化：默认 0.5% 行，±1 步长，并裁剪到 [lo,hi]。
    用于增强匿名性，对统计/相关影响可忽略。
    """
    if col not in df.columns: return
    x = pd.to_numeric(df[col], errors="coerce")
    mask = x.notna()
    pick = (rng.random(len(df)) < frac) & mask
    if not pick.any(): return
    noise = rng.choice(delta_choices, size=pick.sum())
    x.loc[pick] = np.clip((x.loc[pick] + noise).round(), lo, hi)
    df[col] = x

# ===== 你原来的清理函数（保留风格与格式约束）=====
def clean_data_format(df: pd.DataFrame) -> None:
    """确保数据符合 Ci 的字符串整数/一位小数格式"""
    # AGE：若仍有空值，用中位数；最终转整数字符串
    if "AGE" in df.columns:
        age_num = pd.to_numeric(df["AGE"], errors="coerce")
        empty_age_mask = age_num.isna()
        if empty_age_mask.any():
            median_age = age_num.median()
            if pd.isna(median_age): median_age = 45
            df.loc[empty_age_mask, "AGE"] = str(int(median_age))
    numeric_cols = [
        "AGE", "encounter_count", "num_procedures", "num_medications",
        "num_immunizations", "num_allergies", "num_devices",
        "mean_systolic_bp", "mean_diastolic_bp", "mean_weight"
    ]
    float_cols = ["mean_bmi"]
    for col in numeric_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            med = s.median()
            if pd.isna(med): med = 0
            df[col] = s.apply(lambda x: str(int(round(med))) if pd.isna(x) else str(int(round(x))))
    for col in float_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            med = s.median()
            if pd.isna(med): med = 0.0
            df[col] = s.apply(lambda x: f"{med:.1f}" if pd.isna(x) else f"{x:.1f}")
    # 分类变量与 flag 清理
    for col in ["GENDER","RACE","ETHNICITY"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan":"", "None":"", "":""})
            if df[col].isna().any() or (df[col]== "").any():
                mode = df[col].mode()
                if len(mode)>0:
                    fill = mode.iloc[0]
                    df[col] = df[col].replace({"":fill}).fillna(fill)
    for col in ["asthma_flag", "stroke_flag", "obesity_flag", "depression_flag"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan":"","None":"","":""})
            valid_mask = df[col].isin(["0","1"])
            df.loc[~valid_mask, col] = "0"

# ===== main =====
def main():
    parser = argparse.ArgumentParser(description="保相关增强匿名化（最小改动版）")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 统一随机源
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    df = BiDataFrame.read_csv(args.input_csv)
    print(f"原始数据行数: {len(df)}")

    # 1) 分类/flag 规范（不改变取值，仅修空）
    ensure_categoricals(df)
    ensure_flags(df)

    # 2) AGE 派生：仅用于分层的 AGE_BIN（不改变 AGE 值）
    if "AGE" in df.columns:
        df["AGE_BIN"] = _age_bins(df["AGE"])
    else:
        df["AGE_BIN"] = pd.Series(["41-50"]*len(df), index=df.index)

    # 3) 仅对数值列缺失值做分层中位数填补（以 asthma_flag 为核心分层）
    bundle_cols: List[str] = []
    if "AGE" in df.columns: bundle_cols.append("AGE")
    bundle_cols += [c for c in COUNT_COLS if c in df.columns]
    bundle_cols += [c for c in PHYSIO_COLS if c in df.columns]

    hier = build_hierarchy(df)  # 分层从细到粗，始终包含 asthma_flag
    for col in bundle_cols:
        stratified_impute_nan(df, col, hier)

    # 4) 分层“同一排列”的块置换（对 bundle 一起 permute，保持 X|asthma 联合分布）
    if len(bundle_cols) >= 2:
        block_permute_bundle(df, bundle_cols, hier, rng)

    # 5) 单列极轻随机化（可调参数；默认对 num_devices）
    if "num_devices" in df.columns:
        # 边界宽松一些，避免截断影响分布；你也可以改成更小的 frac（如 0.003）
        tiny_randomize_one_count(df, "num_devices", rng, frac=0.005,
                                 delta_choices=(-1, +1), lo=0, hi=100)

    # 6) 删除中间列、做最终格式化
    if "AGE_BIN" in df.columns:
        df.drop(columns=["AGE_BIN"], inplace=True)
    clean_data_format(df)

    # 7) 输出
    Ci_df = CiDataFrame(df)
    Ci_df.to_csv(args.output_csv)
    print(f"✅ 匿名化完成！输出文件: {args.output_csv}")
    print(f"处理后数据行数: {len(df)}")

if __name__ == "__main__":
    main()
