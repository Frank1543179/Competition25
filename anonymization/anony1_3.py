#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强匿名化：
核心方法是按人群分组（“类内”）做高斯 Copula 合成，再把采样到的“分位”（0～1）投影回原列的经验分布
整数列确保为整数
可选再做一次逻辑回归 logit 分布对齐，避免统计漂移
连续体征保留 2 位小数
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm   #提供正态分布的 ppf/cdf
import statsmodels.api as sm   #用于 GLM（逻辑回归）拟合

# -------------------- schema --------------------
#字段“模式”与年龄分箱
#所有数值列（含整数/连续）
NUM_COLS = [
    "AGE","encounter_count","num_procedures","num_medications",
    "num_immunizations","num_allergies","num_devices",
    "mean_systolic_bp","mean_diastolic_bp","mean_bmi","mean_weight"
]
#整数列
INT_COLS = ["AGE","encounter_count","num_procedures","num_medications","num_immunizations","num_allergies","num_devices"]
#类目列
CAT_COLS = ["GENDER","RACE","ETHNICITY"]
#0/1标记列
FLAG_COLS = ["asthma_flag","stroke_flag","obesity_flag","depression_flag"]
#保留两位小数列
CONT_CONTROLS = ["mean_systolic_bp","mean_diastolic_bp","mean_bmi","mean_weight"]  # 两位小数保留列
#年龄段-分组
DEFAULT_BINS = [0,18,45,65,75,200]
DEFAULT_LABELS = ["0-17","18-44","45-64","65-74","75+"]

def agebin(s):
    return pd.cut(pd.to_numeric(s, errors="coerce"),
                  bins=DEFAULT_BINS, right=False,
                  labels=DEFAULT_LABELS, include_lowest=True).astype(str)

# -------------------- utils --------------------
#相关矩阵必须是对称正定才能用来采多元正态，高斯 Copula 合成的“数值安全阀”
def nearest_spd(A, eps=1e-6):
    A = (A + A.T) / 2.0
    vals, vecs = np.linalg.eigh(A)
    vals[vals < eps] = eps
    return (vecs * vals) @ vecs.T

#组内高斯 Copula 合成
def gaussian_copula_group(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, d = values.shape
    X = values.copy().astype(float)
    # 填补缺失：用列中位数
    for j in range(d):
        col = X[:, j]
        m = np.nanmedian(col)
        col[np.isnan(col)] = m
        X[:, j] = col
    # 秩→正态分数
    U = np.zeros_like(X)
    for j in range(d):
        r = pd.Series(X[:, j]).rank(method="average").to_numpy()
        U[:, j] = (r - 0.5) / (n + 1.0)
    Z = norm.ppf(np.clip(U, 1e-6, 1-1e-6))
    # 估计秩相关并采样
    Sigma = np.corrcoef(Z, rowvar=False)
    Sigma = nearest_spd(Sigma)
    Zp = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
    Up = norm.cdf(Zp)
    return Up

#分位投影（整数安全）
def quantile_project_uniforms(Up: np.ndarray, ref_vals: np.ndarray, int_mask: np.ndarray) -> np.ndarray:
    n, d = Up.shape
    Xsyn = np.zeros_like(Up)
    for j in range(d):
        col = pd.to_numeric(ref_vals[:, j], errors="coerce")
        xs = np.sort(col[~np.isnan(col)])
        if len(xs) == 0:
            Xsyn[:, j] = np.nan
            continue
        pos = Up[:, j] * (len(xs)-1)
        if int_mask[j]:
            # 整数列：就近序统计量，避免小数
            idx = np.rint(pos).astype(int)
            idx = np.clip(idx, 0, len(xs)-1)
            Xsyn[:, j] = xs[idx]
        else:
            # 连续列：线性插值
            lo = np.floor(pos).astype(int); hi = np.ceil(pos).astype(int)
            frac = pos - lo
            Xsyn[:, j] = xs[lo]*(1-frac) + xs[hi]*frac
    return Xsyn

# ---------- logistic baseline on Bi (for optional logit alignment) ----------
#逻辑回归基线（可选“logit 对齐”会用）
def build_lr_X(df: pd.DataFrame, target: str):
    y = pd.to_numeric(df[target], errors="coerce")
    X_raw = df.drop(columns=[target], errors="ignore")
    num_cols = [c for c in X_raw.columns if c in NUM_COLS]
    X_num = X_raw[num_cols].apply(pd.to_numeric, errors="coerce") if num_cols else pd.DataFrame(index=df.index)
    cat_cols = [c for c in X_raw.columns if c not in num_cols]
    if cat_cols:
        X_cat = pd.get_dummies(X_raw[cat_cols].replace({"": np.nan}).fillna("(NA)").astype("category"),
                               drop_first=True)
    else:
        X_cat = pd.DataFrame(index=df.index)
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    base_terms = sorted(X.columns.tolist())
    X = X.reindex(columns=base_terms)
    mask = y.notna() & y.isin([0.0, 1.0])
    return X.loc[mask], y.loc[mask].astype(int), base_terms

def build_X_with_terms(df: pd.DataFrame, terms_global: list):
    """按训练的 terms_global 重建设计矩阵；缺失的哑变量列补 0。"""
    X_raw = df.drop(columns=["asthma_flag"], errors="ignore")
    num_in_terms = [c for c in terms_global if c in NUM_COLS]
    X_num = X_raw[num_in_terms].apply(pd.to_numeric, errors="coerce") if num_in_terms else pd.DataFrame(index=df.index)
    X_cat = pd.get_dummies(
        X_raw.drop(columns=num_in_terms, errors="ignore").replace({"": np.nan}).fillna("(NA)").astype("category"),
        drop_first=True
    )
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.reindex(columns=terms_global, fill_value=0.0)
    return X.astype("float64")

def lr_fit_coef_on_bi(df_bi: pd.DataFrame, target="asthma_flag"):
    X, y, terms = build_lr_X(df_bi, target)
    if X.shape[0] < 10 or X.shape[1] == 0:
        return None, None, None, None
    Xc = sm.add_constant(X, has_constant="add").astype("float64")
    model = sm.GLM(y.astype("float64"), Xc, family=sm.families.Binomial())
    res = model.fit()
    coef = res.params.drop(labels=["const"])
    return res, coef, terms, set(NUM_COLS) & set(terms)

# -------------------- main anonymizer --------------------
def anonymize(bi: pd.DataFrame,
              #分组键（年龄段×哮喘标记×性别×种族×族裔）
              group_key=("AGEBIN","asthma_flag","GENDER","RACE","ETHNICITY"),
              k_min=40, seed=7, logit_align=False):
    rng = np.random.default_rng(seed)
    df = bi.copy()

    # 补列
    for c in NUM_COLS + FLAG_COLS + CAT_COLS:
        if c not in df: df[c] = np.nan

    # 低频并类
    def coarsen(df0):
        out = df0.copy()
        out["GENDER"] = out["GENDER"].astype(str).str.upper().str[0].map({"M": "M", "F": "F"}).fillna("U")
        for c in ["RACE","ETHNICITY"]:
            ser = out[c].astype(str).replace({"":"(NA)"})
            vc = ser.value_counts(dropna=False)
            rare = set(vc[vc < 50].index)
            out[c] = ser.apply(lambda v: "other" if v in rare else v)
        return out
    df = coarsen(df)
    df["AGEBIN"] = agebin(df["AGE"])

    # 放宽分组直至所有组 >= k_min
    def relax(keys):
        tiers = [
            list(keys),
            [k for k in keys if k != "ETHNICITY"],
            [k for k in keys if k != "RACE"],
            ["AGEBIN","asthma_flag"]
        ]
        for ks in tiers:
            sizes = df.groupby(ks, sort=False).size()
            if (sizes >= k_min).all():
                return ks
        return ["AGEBIN","asthma_flag"]

    key = relax(list(group_key))
    groups = df.groupby(key, sort=False)

    # 训练 LR（用于可选的 logit 对齐）
    lr_res, lr_coef, lr_terms, _ = lr_fit_coef_on_bi(df, target="asthma_flag")
    exog_names_global = lr_res.model.exog_names if lr_res is not None else None  # 含 const
    terms_global = list(lr_terms) if lr_terms is not None else None              # 不含 const

    # ---- Copula 合成（组内） ----核心生成
    out = df.copy()
    for _, idx in groups.indices.items():
        idl = list(idx)
        sub = df.loc[idl]
        Xref = sub[NUM_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()
        Up = gaussian_copula_group(Xref, rng)
        int_mask = np.array([c in INT_COLS for c in NUM_COLS], dtype=bool)
        Xsyn = quantile_project_uniforms(Up, Xref, int_mask)
        out.loc[idl, NUM_COLS] = Xsyn

    # ---- 数值格式控制：整数→int；连续→2位小数 ----
    for c in INT_COLS:
        ser = pd.to_numeric(out[c], errors="coerce").round()
        ser = ser.clip(lower=0, upper=200) if c == "AGE" else ser.clip(lower=0)
        out[c] = ser.fillna(0).astype("int64")
    for c in [col for col in NUM_COLS if col not in INT_COLS]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    # ---- 可选：logit 对齐（仅沿连续四列方向做最小范数校正） ----
    if logit_align and (lr_res is not None) and (lr_coef is not None) and (exog_names_global is not None) and (terms_global is not None):
        for yv in [0, 1]:
            for ab in DEFAULT_LABELS:
                mask = (df["asthma_flag"] == yv) & (agebin(df["AGE"]) == ab)
                mask = mask.fillna(False)
                ids = out.index[mask]
                if len(ids) == 0:
                    continue

                bi_sub = df.loc[ids]
                ci_sub = out.loc[ids]

                # 按训练模板重建并对齐列
                X_bi = build_X_with_terms(bi_sub, terms_global)
                X_ci = build_X_with_terms(ci_sub, terms_global)
                Xc_bi = sm.add_constant(X_bi, has_constant="add").reindex(columns=exog_names_global, fill_value=0.0)
                Xc_ci = sm.add_constant(X_ci, has_constant="add").reindex(columns=exog_names_global, fill_value=0.0)

                if Xc_bi.shape[0] == 0 or Xc_ci.shape[0] == 0:
                    continue

                # 线性部分（logit）
                logit_bi = lr_res.predict(Xc_bi, which="linear")
                logit_ci = lr_res.predict(Xc_ci, which="linear")

                # 把 Ci 的 logit 秩映射到 Bi 的 logit 分布
                r = pd.Series(logit_ci).rank(method="average").to_numpy()
                r = (r - 1) / max(len(logit_ci) - 1, 1)
                xs = np.sort(logit_bi)
                pos = r * (len(xs) - 1)
                lo = np.floor(pos).astype(int); hi = np.ceil(pos).astype(int); frac = pos - lo
                target_logit = xs[lo] * (1 - frac) + xs[hi] * frac
                delta = target_logit - logit_ci

                # numpy 化，避免 pandas 对多维索引的限制 & 确保形状正确
                logit_ci_np = np.asarray(logit_ci, dtype=float).reshape(-1)  # (n,)
                target_logit_np = np.asarray(target_logit, dtype=float).reshape(-1)  # (n,)
                delta = target_logit_np - logit_ci_np  # (n,)

                # 连续列仅取这四个里训练时真实存在的
                cont_cols = [c for c in ["mean_systolic_bp", "mean_diastolic_bp", "mean_bmi", "mean_weight"] if
                             c in terms_global]
                if not cont_cols:
                    continue

                # 方向向量 & 范数平方
                w_vec = np.asarray([lr_coef.get(c, 0.0) for c in cont_cols], dtype=float)  # (d,)
                norm2 = float(np.dot(w_vec, w_vec))
                if not np.isfinite(norm2) or norm2 < 1e-10:
                    continue

                # 取出 Ci 子集的连续特征矩阵
                idx_ci = X_ci.index  # 与 Xc_ci 行顺序一致
                subX = out.loc[idx_ci, cont_cols].apply(pd.to_numeric, errors="coerce").to_numpy()  # (n,d)

                # 计算最小范数校正： (n,1) @ (1,d) -> (n,d)
                delta_col = delta.reshape(-1, 1)  # (n,1)
                w_row = (w_vec / norm2).reshape(1, -1)  # (1,d)
                adjust = delta_col @ w_row  # (n,d)
                subX2 = subX + adjust  # (n,d)

                # 夹在 Bi 子集分位范围内再回写
                for j, colname in enumerate(cont_cols):
                    ref = pd.to_numeric(bi_sub[colname], errors="coerce")
                    mn, mx = np.nanpercentile(ref, 0.5), np.nanpercentile(ref, 99.5)
                    out.loc[idx_ci, colname] = np.clip(subX2[:, j], mn, mx)

                # 二次小数位控制
                for c in [col for col in NUM_COLS if c not in INT_COLS]:
                    out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    out = out.drop(columns=["AGEBIN"])
    return out

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi", required=True, help="path to Bi.csv")
    ap.add_argument("--out", required=True, help="output Ci.csv")
    ap.add_argument("--group", default="AGEBIN,asthma_flag,GENDER,RACE,ETHNICITY")
    ap.add_argument("--kmin", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--logit-align", action="store_true", help="align Ci logits to Bi along continuous-feature direction")
    args = ap.parse_args()

    df = pd.read_csv(args.bi, dtype=str, keep_default_na=False)
    for c in NUM_COLS + FLAG_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = anonymize(
        df,
        group_key=tuple([s.strip() for s in args.group.split(",") if s.strip()]),
        k_min=args.kmin,
        seed=args.seed,
        logit_align=args.logit_align,
    )
    out.to_csv(args.out, index=False, float_format="%.2f")
    print(f"[OK] Saved to {args.out}")

if __name__ == "__main__":
    main()
