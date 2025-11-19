#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#有用性：哮喘风险因子分析--执行逻辑回归分析，预测asthma_flag（哮喘标志），并输出标准化的结果表格。
"""
哮喘标志的逻辑回归（固定模式输出，方法B）
- 将所有可输出的条目按字典顺序排列，并且使用 --ensure-terms
-将指定的term进行合并，并固定最终输出的顺序（按字典顺序）。
-const 不会输出。
- 输出: AUC 和单个表
 [term, coef, p_value, OR_norm, CI_low_norm, CI_high_norm, VIF_norm]
- OR_norm, CI_*_norm = odds / (1 + odds)
- VIF_norm = 1 - 1 / max(VIF, 1) # VIF=1→0（无相关）, VIF→∞→1（强多重共线性）
- ensure-terms 包含但在学习矩阵中不存在的术语用 NaN 占位。

Logistic Regression for asthma_flag (fixed-schema output, approach B)
- すべての「出力可能な term」を辞書順で並べ、さらに --ensure-terms で
  指定された term を union して最終出力の順序（辞書順）を固定。
- const は出力しない。
- 出力: AUC と単一テーブル
    [term, coef, p_value, OR_norm, CI_low_norm, CI_high_norm, VIF_norm]
- OR_norm, CI_*_norm = odds / (1 + odds)
- VIF_norm = 1 - 1 / max(VIF, 1)    # VIF=1→0（無相関）, VIF→∞→1（強い多重共線性）
- ensure-terms に含まれるが学習行列に存在しない term は NaN で占位。
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm  #统计模型，用于逻辑回归
from sklearn.metrics import roc_auc_score  #计算AUC指标
from sklearn.model_selection import train_test_split  #数据分割为训练集和测试集
from statsmodels.stats.outliers_influence import variance_inflation_factor  #计算方差膨胀因子(VIF)

#检查Series是否为严格的0/1二分类变量
def is_binary_01(s: pd.Series) -> bool:
    if s.empty:
        return False
    ss = s.astype(str).str.strip()
    ss = ss[~ss.str.lower().isin({"nan", "none", ""})]
    if ss.empty:
        return False
    num = pd.to_numeric(ss, errors="coerce")
    if num.isna().any():
        return False
    vals = set(pd.unique(num.dropna().astype(int)))
    return vals.issubset({0, 1}) and len(vals) > 0

#将odds ratio归一化到0-1范围
def odds_to_unit(x):
    arr = np.asarray(x, dtype=float)
    return arr / (1.0 + arr)

#将VIF归一化到0-1范围
def vif_to_unit(v):
    v = np.asarray(v, dtype=float)
    v = np.where(~np.isfinite(v) | (v < 1.0), 1.0, v)
    return 1.0 - 1.0 / v

# ==== 从 1个CSV 创建 LR_asthma 兼容的表格====

COLS_ORDER = ["term", "coef", "p_value", "OR_norm", "CI_low_norm", "CI_high_norm", "VIF_norm"]
#执行逻辑回归分析并返回标准化结果
def run_lr_table(df: pd.DataFrame,            #输入数据框
                 target: str = "asthma_flag", #目标变量名（默认"asthma_flag"）
                 test_size: float = 0.2,      #测试集比例
                 random_state: int = 42,      #随机种子
                 ensure_terms: str = "ETHNICITY_hispanic") -> tuple[pd.DataFrame, float, list[str]]:#确保包含在输出中的term列表
    """
    csv_path を読み、LR_asthma.py と同様に
    - 前処理（数値 + カテゴリone-hot drop_first）
    - ロジスティック回帰（const あり、出力は const 除外）
    - OR/CI を 0-1 化、VIF を 0-1 化
    - 固定順序の term を作る（方式B: 学習に使った列 ∪ ensure_terms を辞書順）
    を行い、[COLS_ORDER] の表を返す。AUC と最終の term リストも返す。
    读取 csv_path，按照 LR_asthma.py 的方式进行：
    - 预处理（数值 类别 one-hot drop_first）
    - 逻辑回归（包含常数项，输出去除常数项）
    - 将 OR/CI 归一化到 0-1，将 VIF 归一化到 0-1
    - 创建固定顺序的项（方式 B：使用过的列与 ensure_terms 进行字典排序的并集）
    并返回 [COLS_ORDER] 表格。同时返回 AUC 和最终的项列表。
    """
    # df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    if target not in df.columns:
        # raise SystemExit(f"[{csv_path}] target column '{target}' not found.")
        raise SystemExit(f"target column '{target}' not found.")
    if not is_binary_01(df[target]):
        # raise SystemExit(f"[{csv_path}] target column '{target}' must be strictly binary 0/1.")
        raise SystemExit(f"target column '{target}' must be strictly binary 0/1.")

    y = pd.to_numeric(df[target], errors="coerce").astype("float64")

    X_raw = df.drop(columns=[target]).copy()
    X_num_try = X_raw.apply(pd.to_numeric, errors="coerce")
    num_cols = [c for c in X_num_try.columns if X_num_try[c].notna().sum() > 0]
    X_num = X_num_try[num_cols] if num_cols else pd.DataFrame(index=df.index)

    cat_cols = [c for c in X_raw.columns if c not in num_cols]
    if cat_cols:
        X_cat = pd.get_dummies(
            X_raw[cat_cols].replace({"": np.nan}).fillna("(NA)").astype("category"),
            drop_first=True
        )
    else:
        X_cat = pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_cat], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        # raise SystemExit(f"[{csv_path}] No usable features after preprocessing.")
        raise SystemExit(f"No usable features after preprocessing.")

    med = X.median(numeric_only=True)
    X = X.fillna(med).astype("float64")
    zero_var = X.nunique(dropna=False) <= 1
    if zero_var.all():
        # raise SystemExit(f"[{csv_path}] All feature columns have zero variance.")
        raise SystemExit(f"All feature columns have zero variance.")
    if zero_var.any():
        X = X.loc[:, ~zero_var]

    base_terms = sorted(X.columns.tolist())
    X = X.reindex(columns=base_terms)

    mask = y.notna() & y.isin([0.0, 1.0])
    X = X.loc[mask]
    y = y.loc[mask]
    if X.shape[0] < 3 or X.shape[1] == 0:
        # raise SystemExit(f"[{csv_path}] Not enough samples or features after cleaning.")
        raise SystemExit(f"Not enough samples or features after cleaning.")

#模型训练和评估
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y.astype(int)
    )
    X_tr_const = sm.add_constant(X_tr, has_constant="add").astype("float64")
    X_te_const = sm.add_constant(X_te, has_constant="add").astype("float64")

    model = sm.GLM(y_tr.astype("float64"), X_tr_const, family=sm.families.Binomial())
    res = model.fit()

    proba = res.predict(X_te_const)
    auc = roc_auc_score(y_te, proba)

    params = res.params.drop(labels=["const"])
    pvals  = res.pvalues.drop(labels=["const"])
    conf   = res.conf_int().drop(index="const")  # columns [0,1]

    OR      = np.exp(params.values)
    CI_low  = np.exp(conf[0].reindex(params.index).values)
    CI_high = np.exp(conf[1].reindex(params.index).values)

    coef_df = pd.DataFrame({
        "term": params.index,
        "coef": params.values,
        "p_value": pvals.values,
        "OR_norm": odds_to_unit(OR),
        "CI_low_norm": odds_to_unit(CI_low),
        "CI_high_norm": odds_to_unit(CI_high),
    })

    # VIF（const除外）
    vif_rows = []
    cols = list(X_tr_const.columns)  # ['const', ...base_terms...]
    for i, col in enumerate(cols):
        if col == "const":
            continue
        try:
            v = float(variance_inflation_factor(X_tr_const.values, i))
        except Exception:
            v = np.nan
        vif_rows.append((col, v))
    vif_df = pd.DataFrame(vif_rows, columns=["term", "VIF"])
    vif_df["VIF_norm"] = vif_to_unit(vif_df["VIF"].values)
    vif_df = vif_df.drop(columns=["VIF"])

    # 固定スキーマ：ensure-terms を union
    ensure_list = [t.strip() for t in ensure_terms.split(",") if t.strip()]
    final_terms = sorted(set(base_terms).union(ensure_list))

    out = (coef_df.merge(vif_df, on="term", how="outer")
                  .set_index("term")
                  .reindex(final_terms)
                  .reset_index())

    # 欠損は 0 に
    for c in COLS_ORDER[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # 列順
    out = out[COLS_ORDER]
    return out, float(auc), final_terms

def main():
    ap = argparse.ArgumentParser(
        description="Logistic regression (asthma_flag) with fixed-schema output and normalized VIF"
    )
    ap.add_argument("csv", help="input CSV")
    ap.add_argument("--target", default="asthma_flag", help="binary target column (default: asthma_flag)")
    ap.add_argument("--test-size", type=float, default=0.2, help="holdout ratio (default: 0.2)")
    ap.add_argument("--random-state", type=int, default=42, help="random seed (default: 42)")
    ap.add_argument(
        "--ensure-terms",
        default="ETHNICITY_hispanic",
        help="カンマ区切りで常に表に載せたい term 名（例: 'ETHNICITY_hispanic,GENDER_M'）"
    )
    args = ap.parse_args()

    # 1) 読み込み＆ターゲット確認
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' not found.")
    if not is_binary_01(df[args.target]):
        raise SystemExit(f"Target column '{args.target}' must be strictly binary 0/1.")
    y = pd.to_numeric(df[args.target], errors="coerce").astype("float64")

    # 2) 特徴量作成：数値 + 文字列はワンホット（drop_first=True）
    X_raw = df.drop(columns=[args.target]).copy()
    X_num_try = X_raw.apply(pd.to_numeric, errors="coerce")
    num_cols = [c for c in X_num_try.columns if X_num_try[c].notna().sum() > 0]
    X_num = X_num_try[num_cols] if num_cols else pd.DataFrame(index=df.index)

    cat_cols = [c for c in X_raw.columns if c not in num_cols]
    if cat_cols:
        X_cat = pd.get_dummies(
            X_raw[cat_cols].replace({"": np.nan}).fillna("(NA)").astype("category"),
            drop_first=True
        )
    else:
        X_cat = pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_cat], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise SystemExit("No usable features after preprocessing.")

    # 3) 欠損補完・ゼロ分散除去
    med = X.median(numeric_only=True)
    X = X.fillna(med).astype("float64")
    zero_var = X.nunique(dropna=False) <= 1
    if zero_var.all():
        raise SystemExit("All feature columns have zero variance.")
    if zero_var.any():
        X = X.loc[:, ~zero_var]

    # 4) 固定順序ベース（現に学習に使う列）
    base_terms = sorted(X.columns.tolist())
    X = X.reindex(columns=base_terms)

    # y と整合
    mask = y.notna() & y.isin([0.0, 1.0])
    X = X.loc[mask]
    y = y.loc[mask]
    if X.shape[0] < 3 or X.shape[1] == 0:
        raise SystemExit("Not enough samples or features after cleaning.")

    # 5) 学習・評価
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y.astype(int)
    )
    X_tr_const = sm.add_constant(X_tr, has_constant="add").astype("float64")
    X_te_const = sm.add_constant(X_te, has_constant="add").astype("float64")
    model = sm.GLM(y_tr.astype("float64"), X_tr_const, family=sm.families.Binomial())
    res = model.fit()

    proba = res.predict(X_te_const)
    auc = roc_auc_score(y_te, proba)

    # 6) 係数・p・CI → OR系を0-1化（const除外）
    params = res.params.drop(labels=["const"])
    pvals  = res.pvalues.drop(labels=["const"])
    conf   = res.conf_int().drop(index="const")  # columns: [0,1] (low, high)

    OR      = np.exp(params.values)
    CI_low  = np.exp(conf[0].reindex(params.index).values)
    CI_high = np.exp(conf[1].reindex(params.index).values)

    coef_df = pd.DataFrame({
        "term": params.index,
        "coef": params.values,
        "p_value": pvals.values,
        "OR_norm": odds_to_unit(OR),
        "CI_low_norm": odds_to_unit(CI_low),
        "CI_high_norm": odds_to_unit(CI_high),
    })

    # 7) VIF（const除外）→ 正規化のみ出力
    vif_rows = []
    cols = list(X_tr_const.columns)  # ['const', ...base_terms...]
    for i, col in enumerate(cols):
        if col == "const":
            continue
        try:
            v = float(variance_inflation_factor(X_tr_const.values, i))
        except Exception:
            v = np.nan
        vif_rows.append((col, v))
    vif_df = pd.DataFrame(vif_rows, columns=["term", "VIF"])
    vif_df["VIF_norm"] = vif_to_unit(vif_df["VIF"].values)
    vif_df = vif_df.drop(columns=["VIF"])

    # 8) ★固定スキーマ（方式B）：ensure-terms を union し、辞書順で最終順序を固定
    ensure_list = []
    if args.ensure_terms:
        ensure_list = [t.strip() for t in args.ensure_terms.split(",") if t.strip()]
    final_terms = sorted(set(base_terms).union(ensure_list))

    # 係数表とVIF_normを term で結合し、final_terms で reindex（存在しないtermはNaNで占位）
    out = (coef_df.merge(vif_df, on="term", how="outer")
                  .set_index("term")
                  .reindex(final_terms)
                  .reset_index())

    # 9) 出力
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print(f"\nAUC (holdout): {auc:.6f}")
    print("\n=== Logistic regression summary (const excluded; fixed-term schema with ensure-terms) ===")
    with pd.option_context('display.float_format', lambda x: f"{x:.6g}"):
        # 列順を固定
        cols_order = ["term", "coef", "p_value", "OR_norm", "CI_low_norm", "CI_high_norm", "VIF_norm"]
        print(out[cols_order].to_string(index=False))

if __name__ == "__main__":
    main()
