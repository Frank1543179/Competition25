#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anonymize_ci.py
- 读入 Bi.csv，输出 Ci.csv
- 两种模式：
  * utility（默认）：分层联合置乱 + 极小去重噪声 + 轻量分位对齐（保 stats/KW/LR 稳定）
  * privacy：    分层联合置乱 + 组内凸组合 + 中等噪声 + 分位对齐（更抗攻击）
- 分层键建议：AGEBIN×asthma_flag（可加 obesity_flag）
- 评估：用 eval_all.py 计算 Ci_utility（三项 diff 越小越好） [权重 40/20/20]
"""

import argparse, numpy as np, pandas as pd

NUM_COLS = [
    "AGE","encounter_count","num_procedures","num_medications",
    "num_immunizations","num_allergies","num_devices",
    "mean_systolic_bp","mean_diastolic_bp","mean_bmi","mean_weight"
]
# 需要强制为整数输出的列：
INT_COLS = [
    "AGE","encounter_count","num_procedures","num_medications",
    "num_immunizations","num_allergies","num_devices"
]
CAT_COLS = ["GENDER","RACE","ETHNICITY"]
FLAG_COLS = ["asthma_flag","stroke_flag","obesity_flag","depression_flag"]

DEFAULT_BINS = [0,18,45,65,75,200]
DEFAULT_LABELS = ["0-17","18-44","45-64","65-74","75+"]

def agebin(s):
    return pd.cut(pd.to_numeric(s, errors="coerce"),
                  bins=DEFAULT_BINS, right=False,
                  labels=DEFAULT_LABELS, include_lowest=True).astype(str)

def coarsen_cats(df, min_count=50):
    out = df.copy()
    out["GENDER"] = out["GENDER"].astype(str).str.upper().str[0].map({"M":"M","F":"F"}).fillna("U")
    for c in ["RACE","ETHNICITY"]:
        ser = out[c].astype(str).replace({"":"(NA)"})
        vc = ser.value_counts(dropna=False)
        rare = set(vc[vc < min_count].index)
        out[c] = ser.apply(lambda v: "other" if v in rare else v)
    return out

def grouped_indices(df, key_cols):
    g = df.groupby(key_cols, sort=False).indices
    return {k:list(idx) for k, idx in g.items()}

def joint_permute_numeric(df, group_index, max_frac=0.15, rng=None):
    rng = np.random.default_rng(rng)
    X = df[NUM_COLS].apply(pd.to_numeric, errors="coerce").copy()
    for _, idx in group_index.items():
        n = len(idx)
        if n <= 1: continue
        w = max(3, int(max_frac*n))
        base = np.arange(n)
        shift = rng.integers(1, min(w, n))
        perm = np.roll(base, shift)
        # 轻量随机互换
        for r in range(n):
            if rng.random() < 0.2:
                a = r
                b = rng.integers(max(0,r-w), min(n-1,r+w)+1)
                perm[a], perm[b] = perm[b], perm[a]
        X.loc[idx] = X.loc[idx].iloc[perm].values
    return X

def convex_mix_block(X, group_index, k_mix=3, noise_frac=0.08, rng=None):
    rng = np.random.default_rng(rng)
    X2 = X.copy()
    for _, idx in group_index.items():
        A = X.loc[idx].to_numpy(dtype=float)
        n, d = A.shape
        if n == 0: continue
        std = np.nanstd(A, axis=0)
        for i, rid in enumerate(idx):
            chose = np.arange(n) if n <= k_mix else rng.choice(n, size=k_mix, replace=False)
            w = rng.random(len(chose)); w = w/w.sum()
            mixed = (A[chose]*w[:,None]).sum(axis=0)
            noise = rng.normal(0.0, noise_frac*std, size=d)
            X2.loc[rid] = mixed + noise
    return X2

def quantile_match(new_col, src_col, group_index, integer=False):
    """
    若 integer=True：使用“阶梯式”秩映射（就近取序统计量），保持输出取值来自源分布（对整数列无小数）。
    若 integer=False：保留原来的线性插值映射。
    """
    out = new_col.copy()
    for _, idx in group_index.items():
        x_new = pd.to_numeric(new_col.loc[idx], errors="coerce").to_numpy()
        x_src = pd.to_numeric(src_col.loc[idx], errors="coerce").to_numpy()
        # 去掉 NaN 的源值以避免传播 NaN
        xs = x_src[~np.isnan(x_src)]
        if len(x_new) <= 1 or len(xs) == 0:
            out.loc[idx] = x_src  # 退化情形
            continue
        xs = np.sort(xs)
        r = pd.Series(x_new).rank(method="average").to_numpy()
        r = (r-1)/(len(x_new)-1)

        pos = r*(len(xs)-1)
        if integer:
            # 整数列：就近取序统计位置（避免插值产生小数）
            ind = np.rint(pos).astype(int)
            ind = np.clip(ind, 0, len(xs)-1)
            mapped = xs[ind]
        else:
            # 连续列：线性插值（更贴近分位）
            lo = np.floor(pos).astype(int); hi = np.ceil(pos).astype(int)
            frac = pos - lo
            mapped = xs[lo]*(1-frac) + xs[hi]*frac

        out.loc[idx] = mapped
    return out

def anonymize(df, mode="utility",
              group_key=("AGEBIN","asthma_flag"),
              k_min=25, max_swap_frac=0.12,
              tiny_noise=0.02,      # utility 模式噪声比例（相对组内std）
              mix_noise=0.08,       # privacy 模式噪声比例
              k_mix=3, seed=42):
    df = df.copy()
    # 标准化与分层键
    for c in NUM_COLS + FLAG_COLS + CAT_COLS:
        if c not in df: df[c] = np.nan
    df = coarsen_cats(df, min_count=50)
    df["AGEBIN"] = agebin(df["AGE"])
    # 组键解析
    key = [("AGEBIN" if c=="AGEBIN" else c) for c in group_key]

    # 若某些小组 < k_min，可自动放宽：优先丢弃 obesity_flag -> 丢弃 ETHNICITY -> 丢弃 RACE
    def relax(keys):
        tiers = [
            keys,
            [k for k in keys if k != "obesity_flag"],
            [k for k in keys if k != "ETHNICITY"],
            [k for k in keys if k != "RACE"],
            ["AGEBIN","asthma_flag"]
        ]
        for ks in tiers:
            sizes = df.groupby(ks, sort=False).size()
            if (sizes >= k_min).all(): return ks
        return ["AGEBIN","asthma_flag"]

    key = relax(key)
    gindex = grouped_indices(df, key)

    # 数值块——组内“联合”操作（保持列间相关结构）
    X_orig = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    X_perm = joint_permute_numeric(df, gindex, max_frac=max_swap_frac, rng=seed)

    if mode == "utility":
        # 极小去重噪声
        X = X_perm.copy()
        for _, idx in gindex.items():
            sub = X.loc[idx]
            std = sub.std(numeric_only=True).replace(0.0, 1.0)
            noise = np.random.default_rng(seed).normal(0.0, tiny_noise, size=sub.shape) * std.values
            X.loc[idx] = sub + noise
        # 分位对齐：整数列用阶梯式映射，其它列用线性插值
        X_matched = X.copy()
        for c in NUM_COLS:
            X_matched[c] = quantile_match(
                X[c], X_orig[c], gindex,
                integer=(c in INT_COLS)
            )
    else:  # privacy
        X_mix = convex_mix_block(X_perm, gindex, k_mix=k_mix, noise_frac=mix_noise, rng=seed)
        X_matched = X_mix.copy()
        for c in NUM_COLS:
            X_matched[c] = quantile_match(
                X_mix[c], X_orig[c], gindex,
                integer=(c in INT_COLS)
            )

    out = df.copy()
    out[NUM_COLS] = X_matched

    # 清理临时列
    out = out.drop(columns=["AGEBIN"])

    # —— 输出前的数值格式控制 ——
    # 1) 整数列：四舍五入→裁剪（非负；AGE 限定 0~200）→转 int
    for c in INT_COLS:
        if c in out.columns:
            series = pd.to_numeric(out[c], errors="coerce")
            if c == "AGE":
                series = series.round().clip(lower=0, upper=200)
            else:
                series = series.round().clip(lower=0)
            # 若仍有 NaN，用 0 补齐（避免 CSV 写出为空）
            series = series.fillna(0).astype("int64")
            out[c] = series

    # 2) 非整数的数值列：保留 3 位小数
    for c in [col for col in NUM_COLS if col not in INT_COLS]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(3)

    return out, {"final_group_key": key, "num_groups": len(gindex)}

def main():
    ap = argparse.ArgumentParser()
    # 位置参数：兼容 “script BI OUT” 的写法
    ap.add_argument("bi_pos", nargs="?", help="Bi.csv path (positional)")
    ap.add_argument("out_pos", nargs="?", help="Ci.csv output path (positional)")
    # 旗标参数（可选）
    ap.add_argument("--bi", help="path to Bi.csv")
    ap.add_argument("--out", default=None, help="output Ci.csv")  # 默认先不给值
    ap.add_argument("--mode", choices=["utility", "privacy"], default="utility")
    ap.add_argument("--group", default="AGEBIN,asthma_flag", help="group key, comma-separated")
    ap.add_argument("--kmin", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 解析优先级：位置参数 > 旗标 > 默认 "Ci.csv"
    bi_path = args.bi_pos if args.bi_pos else (args.bi if args.bi else None)
    out_path = args.out_pos if args.out_pos else (args.out if args.out else "Ci.csv")

    if not bi_path:
        ap.error("需要提供 Bi 数据路径：作为第一个位置参数，或使用 --bi <path>。")

    df = pd.read_csv(bi_path, dtype=str, keep_default_na=False)

    # 强制必要列的数值类型（读入阶段先转为数值，便于后续计算）
    for c in NUM_COLS + FLAG_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out, info = anonymize(
        df, mode=args.mode,
        group_key=tuple([s.strip() for s in args.group.split(",") if s.strip()]),
        k_min=args.kmin, seed=args.seed
    )

    out.to_csv(out_path, index=False)
    print(f"[OK] Saved to {out_path}. Final group key = {info['final_group_key']} (groups={info['num_groups']})")

if __name__ == "__main__":
    main()
