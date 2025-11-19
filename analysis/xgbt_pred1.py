#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使用XGBoost进行二分类预测（版本校验 + 特征对齐 + 概率校准/削顶）
"""
xgbt_pred.py  — version check & robust schema (+ calibration & clipping)

- 读取 Booster JSON 并校验 xgboost 版本（运行时版本必须 >= 模型版本）
- attributes（最优先）/ learner.attributes / learner.feature_names 里提取 feature_names、target
- 依据 feature_names 对 test 进行列对齐（缺列补0、冗余丢弃），顺序一致
- 预测时：
    * 先拿 margin（output_margin=True）
    * 若模型 attributes 含 "platt_a/platt_b"，做 Platt 缩放；否则若含 "temp_scaling_T"，做温度缩放；否则普通 sigmoid
    * 概率削顶（clip到[eps, 1-eps]，默认 eps=0.02，可用 --clip-eps 调整/关闭）
- 输出 Accuracy、AUC、LogLoss
"""

import argparse
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import List, Optional, Tuple
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

# --------- version utils ---------
def ver_tuple_from_runtime(v: str) -> Tuple[int, int, int]:
    parts = []
    for tok in v.split("."):
        num = ""
        for ch in tok:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            parts.append(int(num))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

def ver_tuple_from_model(v) -> Tuple[int, int, int]:
    if isinstance(v, list) and len(v) >= 3:
        try:
            return (int(v[0]), int(v[1]), int(v[2]))
        except Exception:
            return (0, 0, 0)
    return (0, 0, 0)

# --------- schema extraction ---------
def extract_schema_flex(j: dict, cli_target: Optional[str]) -> tuple[Optional[List[str]], str]:
    attrs = j.get("attributes")
    if not isinstance(attrs, dict):
        attrs = j.get("learner", {}).get("attributes", {}) or {}

    feat: Optional[List[str]] = None
    fn_val = attrs.get("feature_names", None)
    if isinstance(fn_val, str):
        try:
            tmp = json.loads(fn_val)
            if isinstance(tmp, list) and all(isinstance(x, str) for x in tmp):
                feat = tmp
        except Exception:
            pass
    elif isinstance(fn_val, list) and all(isinstance(x, str) for x in fn_val):
        feat = fn_val
    if feat is None:
        lf = j.get("learner", {}).get("feature_names")
        if isinstance(lf, list) and all(isinstance(x, str) for x in lf) and len(lf) > 0:
            feat = lf

    target = cli_target or attrs.get("target") or "stroke_flag"
    return feat, target

# --------- preprocessing ---------
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

def build_X(df: pd.DataFrame, target: Optional[str]) -> pd.DataFrame:
    cols_drop = [c for c in [target] if c and c in df.columns]
    X_raw = df.drop(columns=cols_drop, errors="ignore").copy()

    X_num_try = X_raw.apply(pd.to_numeric, errors="coerce")
    num_cols = [c for c in X_num_try.columns if X_num_try[c].notna().sum() > 0]
    X_num = X_num_try[num_cols] if num_cols else pd.DataFrame(index=df.index)

    cat_cols = [c for c in X_raw.columns if c not in num_cols]
    X_cat = (pd.get_dummies(X_raw[cat_cols].astype("category"), drop_first=True)
             if cat_cols else pd.DataFrame(index=df.index))

    X = pd.concat([X_num, X_cat], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise SystemExit("No usable features after preprocessing.")

    zero_var = X.nunique(dropna=False) <= 1
    if zero_var.all():
        raise SystemExit("All feature columns have zero variance.")
    if zero_var.any():
        X = X.loc[:, ~zero_var]

    X = X.reindex(columns=sorted(X.columns))
    return X.astype("float32")

# --------- main ---------
def main():
    ap = argparse.ArgumentParser(description="Predict with trained XGBoost JSON model (version-checked; robust schema; calibrated).")
    ap.add_argument("model_json", help="trained model JSON (Booster.save_model)")
    ap.add_argument("--test-csv", required=True, help="test CSV with header")
    ap.add_argument("--target", default=None, help="target column name (override model metadata)")
    ap.add_argument("--threshold", type=float, default=0.5, help="probability threshold for class 1 (default: 0.5)")
    ap.add_argument("--clip-eps", type=float, default=0.02, help="probability clipping epsilon in [0,0.2] (0=off)")
    args = ap.parse_args()

    # 读取模型 JSON 做版本校验 & 提特征名/目标
    with open(args.model_json, "r", encoding="utf-8") as f:
        j = json.load(f)
    model_ver = ver_tuple_from_model(j.get("version"))
    runtime_ver = ver_tuple_from_runtime(xgb.__version__)
    if model_ver > runtime_ver:
        mv = ".".join(map(str, model_ver))
        rv = ".".join(map(str, runtime_ver))
        raise SystemExit(
            f"xgboost runtime ({rv}) はモデルのバージョン ({mv}) より古く、JSON を読み込めません。\n"
            f"→ `python3 -m pip install --user --upgrade \"xgboost=={mv}\"` で更新するか、"
            f"同じ/古いバージョンでモデルを書き出してください。"
        )

    feature_names_model, target = extract_schema_flex(j, args.target)

    # 读取测试数据并校验目标列
    df_te = pd.read_csv(args.test_csv, dtype=str, keep_default_na=False)
    if target not in df_te.columns:
        raise SystemExit(f"Target column '{target}' not found in test CSV.")
    if not is_binary_01(df_te[target]):
        raise SystemExit(f"Target column '{target}' must be strictly binary 0/1 in test CSV.")
    y_true = pd.to_numeric(df_te[target], errors="coerce").astype(int).values

    X_te = build_X(df_te, target=target)
    if feature_names_model:
        for col in feature_names_model:
            if col not in X_te.columns:
                X_te[col] = 0.0
        X_te = X_te.reindex(columns=feature_names_model)

    # 加载 Booster 并做预测（margin -> 概率），支持校准与削顶
    booster = xgb.Booster()
    booster.load_model(args.model_json)

    # 构造 DMatrix
    dtest = xgb.DMatrix(X_te, feature_names=feature_names_model if feature_names_model else None)

    # 先拿 margin
    margins = booster.predict(dtest, validate_features=True, output_margin=True)

    # 读取校准参数（若存在）
    attrs = booster.attributes() if hasattr(booster, "attributes") else {}
    platt_a = attrs.get("platt_a", None)
    platt_b = attrs.get("platt_b", None)
    temp_T  = attrs.get("temp_scaling_T", None)

    # margin -> prob
    if platt_a is not None and platt_b is not None:
        try:
            a = float(platt_a); b = float(platt_b)
            proba = 1.0 / (1.0 + np.exp(-(a * margins + b)))
            print(f"[info] Using Platt scaling: a={a:.6f}, b={b:.6f}")
        except Exception:
            proba = 1.0 / (1.0 + np.exp(-margins))
            print("[warn] Invalid platt_a/platt_b; fallback to sigmoid.")
    elif temp_T is not None:
        try:
            T = max(1e-6, float(temp_T))
            proba = 1.0 / (1.0 + np.exp(-(margins / T)))
            print(f"[info] Using temperature scaling: T={T:.6f}")
        except Exception:
            proba = 1.0 / (1.0 + np.exp(-margins))
            print("[warn] Invalid temp_scaling_T; fallback to sigmoid.")
    else:
        proba = 1.0 / (1.0 + np.exp(-margins))

    # 概率削顶（进一步抑制过度自信；不改变排序）
    eps = float(args.clip_eps)
    if eps > 0:
        eps = max(0.0, min(0.2, eps))
        proba = np.clip(proba, eps, 1.0 - eps)
        print(f"[info] Probability clipping applied: eps={eps:.3f}")

    # 评估
    if proba.shape[0] != y_true.shape[0]:
        raise SystemExit("Prediction length mismatch with ground truth length.")
    y_pred = (proba >= float(args.threshold)).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, proba)
    except Exception:
        auc = float("nan")
    try:
        nll = log_loss(y_true, proba, labels=[0, 1])
    except Exception:
        nll = float("nan")

    print(f"Accuracy (threshold={args.threshold:.3f}): {acc:.6f}")
    print(f"AUC: {auc:.6f}   LogLoss: {nll:.6f}")

if __name__ == "__main__":
    main()
