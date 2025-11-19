#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#创建匿名化数据 CCi.csv 的示例代码
#按照“逐行修改”随机化处理原始数据得到新的匿名数据
import sys, os
import argparse
import random
from typing import List

import numpy as np
import pandas as pd

# モジュールの相対参照制限を強制的に回避
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'util'))
from pws_data_format  import BiDataFrame, CiDataFrame

def mutate_categorical(series: pd.Series, p: float) -> pd.Series:
    """仅替换非空单元格，以概率 p 替换为列中的“其他值”。"""
    s = series.astype(str)
    uniq = [v for v in sorted(set(s) - {""})]
    if len(uniq) < 2:
        return s  # 置換不能
    mask = (s != "") & (np.random.rand(len(s)) < p)

    def pick_other(val: str) -> str:
        choices = [u for u in uniq if u != val]
        return random.choice(choices) if choices else val

    s.loc[mask] = s.loc[mask].map(pick_other)
    return s


def process_int_column(df: pd.DataFrame, col: str, lo: int, hi: int) -> None:
    """整数列：记录空白个数→非空添加随机数并限制范围→用范围填充空白→最后随机使原来的空白个数变为空白。"""
    if col not in df.columns:
        return
    s_raw = df[col].astype(str)
    is_blank = s_raw.str.strip().eq("")
    blanks_n = int(is_blank.sum())

    s_num = pd.to_numeric(s_raw.where(~is_blank, np.nan), errors="coerce")
    non_na = s_num.dropna()
    if non_na.empty:
        df[col] = s_raw
        return

    vmin = int(np.floor(non_na.min()))
    vmax = int(np.ceil(non_na.max()))

    delta = np.random.randint(lo, hi + 1, size=len(s_num))
    s_num = s_num.add(delta).clip(lower=vmin, upper=vmax)

    fill_vals = np.random.randint(vmin, vmax + 1, size=len(s_num))
    s_num = s_num.where(~is_blank, fill_vals)

    out = s_num.round(0).astype(int).astype(str)

    if blanks_n > 0:
        idx = np.random.choice(len(out), size=blanks_n, replace=False)
        out.iloc[idx] = ""
    df[col] = out


def process_float_add(df: pd.DataFrame, col: str, lo: float, hi: float, decimals: int = 2) -> None:
    """浮动小数点列：非空则加乱数并Clamp。空白保持不变。"""
    if col not in df.columns:
        return
    s_raw = df[col].astype(str)
    is_blank = s_raw.str.strip().eq("")
    s_num = pd.to_numeric(s_raw.where(~is_blank, np.nan), errors="coerce")
    non_na = s_num.dropna()
    if non_na.empty:
        return

    vmin = float(non_na.min())
    vmax = float(non_na.max())
    delta = np.random.uniform(lo, hi, size=len(s_num))
    s_num = s_num.add(delta).clip(lower=vmin, upper=vmax).round(decimals)

    df[col] = s_num.where(~is_blank, "").astype(object)


def process_float_with_blanks(df: pd.DataFrame, col: str, lo: float, hi: float, decimals: int = 2) -> None:
    """浮动小数点列：保存空白个数→变为非空并加入噪声→限制→用[min,max]填充空白→最后随机将原空白个数变为空白。"""
    if col not in df.columns:
        return
    s_raw = df[col].astype(str)
    is_blank = s_raw.str.strip().eq("")
    blanks_n = int(is_blank.sum())

    s_num = pd.to_numeric(s_raw.where(~is_blank, np.nan), errors="coerce")
    non_na = s_num.dropna()
    if non_na.empty:
        return

    vmin = float(non_na.min())
    vmax = float(non_na.max())
    delta = np.random.uniform(lo, hi, size=len(s_num))
    s_num = s_num.add(delta).clip(lower=vmin, upper=vmax)

    fill_vals = np.random.uniform(vmin, vmax, size=len(s_num))
    s_num = s_num.where(~is_blank, fill_vals).round(decimals)

    out = s_num.astype(str)
    if blanks_n > 0:
        idx = np.random.choice(len(out), size=blanks_n, replace=False)
        out.iloc[idx] = ""
    df[col] = out


def flip_flag_with_prob(df: pd.DataFrame, col: str, p: float) -> None:
    """以概率 p 反转 0/1 标志。仅针对非空和 0/1。"""
    if col not in df.columns:
        return
    s_raw = df[col].astype(str)
    is_zero = s_raw == "0"
    is_one = s_raw == "1"
    mask = (is_zero | is_one) & (np.random.rand(len(s_raw)) < p)

    flipped = s_raw.copy()
    flipped.loc[mask & is_zero] = "1"
    flipped.loc[mask & is_one] = "0"
    df[col] = flipped


def process_age_add(df: pd.DataFrame, col: str = "AGE",
                    lo: int = -2, hi: int = 2,
                    min_age: int = 2, max_age: int = 110) -> None:
    """AGE列：仅对非空的整数添加噪声（[lo,hi]），并在[min_age,max_age]范围内进行限制。空白保持不变。"""
    if col not in df.columns:
        return
    s_raw = df[col].astype(str)
    is_blank = s_raw.str.strip().eq("")
    s_num = pd.to_numeric(s_raw.where(~is_blank, np.nan), errors="coerce")
    non_na = s_num.dropna()
    if non_na.empty:
        return

    delta = np.random.randint(lo, hi + 1, size=len(s_num))
    s_num = s_num.add(delta).clip(lower=min_age, upper=max_age).round(0)

    # 出力は他列と同様に文字列
    df[col] = s_num.where(~is_blank, "").astype(object).astype(str)


def main():
    parser = argparse.ArgumentParser(description="匿名化します（BIRTHDATE等の日付処理なし）。")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード（再現用、省略可）")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Biを読み込み
    df = BiDataFrame.read_csv(args.input_csv)

    # ---- カテゴリ列のランダム置換 ----
    if "GENDER" in df.columns:
        df["GENDER"] = mutate_categorical(df["GENDER"], p=0.04)
    if "RACE" in df.columns:
        df["RACE"] = mutate_categorical(df["RACE"], p=0.04)
    if "ETHNICITY" in df.columns:
        df["ETHNICITY"] = mutate_categorical(df["ETHNICITY"], p=0.03)
    # if "GENDER" in df.columns:
    #     df["GENDER"] = mutate_categorical(df["GENDER"], p=0.11)---5  调低到 0.05 左右，类别分布太偏移会影响回归的dummy变量
    # if "RACE" in df.columns:
    #     df["RACE"] = mutate_categorical(df["RACE"], p=0.12)
    # if "ETHNICITY" in df.columns:
    #     df["ETHNICITY"] = mutate_categorical(df["ETHNICITY"], p=0.13)

    # ---- AGE（整数ノイズ; 空欄はそのまま）----
    process_age_add(df, col="AGE", lo=-2, hi=2, min_age=2, max_age=110)

    # ---- 整数ノイズ付加 ----
    process_int_column(df, "encounter_count",  lo=-10, hi=10)
    process_int_column(df, "num_procedures",   lo=-10, hi=10)
    process_int_column(df, "num_medications",  lo=-5,  hi=5)
    process_int_column(df, "num_immunizations",lo=-1,  hi=1)
    # process_int_column(df, "num_immunizations", lo=-3, hi=3)---2
    process_int_column(df, "num_allergies",    lo=-1,  hi=1)
    process_int_column(df, "num_devices", lo=-2, hi=2)
    # process_int_column(df, "num_devices",      lo=-5,  hi=5)---1



    # ---- *_flag は確率で反転 ----
    flip_flag_with_prob(df, "asthma_flag",     p=0.03)
    flip_flag_with_prob(df, "stroke_flag",     p=0.03)
    flip_flag_with_prob(df, "obesity_flag",    p=0.03)
    flip_flag_with_prob(df, "depression_flag", p=0.05)
    # flip_flag_with_prob(df, "asthma_flag", p=0.14)---4   0.14 ~ 0.17反转概率设置的太高
    # flip_flag_with_prob(df, "stroke_flag", p=0.15)
    # flip_flag_with_prob(df, "obesity_flag", p=0.16)
    # flip_flag_with_prob(df, "depression_flag", p=0.17)

    # ---- 実数ノイズ付加 ----
    process_float_add(df, "mean_systolic_bp",   lo=-5.0, hi=5.0, decimals=2)
    # process_float_add(df, "mean_systolic_bp", lo=-10.0, hi=10.0, decimals=2)---3
    process_float_add(df, "mean_diastolic_bp",  lo=-4.0,  hi=4.0,  decimals=2)
    # process_float_add(df, "mean_diastolic_bp", lo=-8.0, hi=8.0, decimals=2)
    process_float_add(df, "mean_weight",        lo=-3.0,  hi=3.0,  decimals=2)

    # ---- 実数ノイズ付加（空欄処理込み) ----
    process_float_with_blanks(df, "mean_bmi",   lo=-6.0,  hi=6.0,  decimals=2)

    # ---- 出力 ----
    Ci_df = CiDataFrame(df)
    Ci_df.to_csv(args.output_csv)


if __name__ == "__main__":
    main()
