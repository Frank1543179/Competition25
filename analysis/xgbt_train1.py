#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xgbt_train.py  (contest-spec compliant; no post-edit)

- 读取 CSV（无缺失、无ID），数值化 + one-hot(drop_first=True) + 去零方差 + 列名升序固定
- XGBoost（二分类）训练：强正则 + 子采样 + 以 logloss 早停
- 标签平滑 label smoothing（降低过度自信）
- 类不平衡自动 scale_pos_weight = neg/pos
- 可选校准：温度缩放 or Platt，写入 Booster attributes
- 保存 JSON 前仅 set_attr，不做后编辑
"""

import argparse
import json
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import xgboost as xgb


def is_binary_01(s: pd.Series) -> bool:
    if s.empty:
        return False
    ss = s.astype(str).str.strip()
    if (ss == "").any():
        return False
    num = pd.to_numeric(ss, errors="coerce")
    if num.isna().any():
        return False
    vals = set(pd.unique(num.astype(int)))
    return vals.issubset({0, 1}) and len(vals) > 0


def build_X(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        raise SystemExit(f"Target column '{target}' not found in CSV.")
    X_raw = df.drop(columns=[target]).copy()

    # 尝试数值化
    X_num_try = X_raw.apply(pd.to_numeric, errors="coerce")
    num_cols = [c for c in X_num_try.columns if X_num_try[c].notna().sum() > 0]
    X_num = X_num_try[num_cols] if num_cols else pd.DataFrame(index=df.index)

    # 其余作为类别做 one-hot
    cat_cols = [c for c in X_raw.columns if c not in num_cols]
    X_cat = pd.get_dummies(
        X_raw[cat_cols].astype("category"), drop_first=True
    ) if cat_cols else pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_cat], axis=1)

    # 去零方差
    zero_var = X.nunique(dropna=False) <= 1
    if zero_var.all():
        raise SystemExit("All feature columns have zero variance.")
    if zero_var.any():
        X = X.loc[:, ~zero_var]

    # 列名升序 & 类型
    X = X.reindex(columns=sorted(X.columns)).astype("float32")
    return X


def main():
    ap = argparse.ArgumentParser(description="Train XGBoost binary classifier and export JSON (no post-edit).")
    ap.add_argument("train_csv", help="training CSV with header")
    ap.add_argument("--model-json", required=True, help="output model JSON path")
    ap.add_argument("--target", default="stroke_flag", help="binary target column (default: stroke_flag)")
    ap.add_argument("--test-size", type=float, default=0.1, help="validation ratio (default: 0.1)")

    # 更保守的默认超参（可覆盖）
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--min-child-weight", type=float, default=8.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--reg-alpha", type=float, default=2.0)
    ap.add_argument("--reg-lambda", type=float, default=8.0)
    ap.add_argument("--subsample", type=float, default=0.7)
    ap.add_argument("--colsample-bytree", type=float, default=0.7)
    ap.add_argument("--max-delta-step", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)
    ap.add_argument("--label-smooth", type=float, default=0.03, help="epsilon in [0,0.1]")
    ap.add_argument("--seed", type=int, default=7)

    # 校准方式：none / temp / platt
    ap.add_argument("--calib", choices=["none", "temp", "platt"], default="none",
                    help="probability calibration written into Booster attributes (default: none)")

    args = ap.parse_args()

    # 读入
    df = pd.read_csv(args.train_csv, dtype=str, keep_default_na=False)

    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' not found.")
    if not is_binary_01(df[args.target]):
        raise SystemExit(f"Target column '{args.target}' must be strictly binary 0/1.")

    y = pd.to_numeric(df[args.target], errors="coerce").astype(int).values
    X = build_X(df, target=args.target)
    feature_names: List[str] = list(X.columns)

    # 拆分
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # 类不平衡权重
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    # 标签平滑（只作用于训练，验证保持真实标签）
    eps = max(0.0, min(0.1, float(args.label_smooth)))
    y_tr_smooth = y_tr * (1.0 - 2.0 * eps) + eps

    # ===== 用原生 xgboost.train 以支持 soft label =====
    dtrain = xgb.DMatrix(X_tr, label=y_tr_smooth, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "tree_method": "hist",
        "eta": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "gamma": args.gamma,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "subsample": min(args.subsample, 0.9),
        "colsample_bytree": min(args.colsample_bytree, 0.9),
        "max_delta_step": args.max_delta_step,
        "seed": args.seed,
        "nthread": -1,
        "scale_pos_weight": (float((y_tr == 0).sum()) / float((y_tr == 1).sum())) if (y_tr == 1).sum() > 0 else 1.0,
    }

    watchlist = [(dtrain, "train"), (dval, "val")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        evals=watchlist,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=False,
    )

    # 验证集评估（用概率）
    proba_val = booster.predict(dval)
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
    y_pred = (proba_val >= 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, proba_val)
    nll = log_loss(y_val, proba_val, labels=[0, 1])
    print(f"Validation: ACC={acc:.6f}  AUC={auc:.6f}  LOGLOSS={nll:.6f}")

    # 保存必要 attributes（不后编辑 JSON）
    booster.set_attr(feature_names=json.dumps(feature_names, ensure_ascii=False))
    booster.set_attr(target=args.target)
    booster.set_attr(xgboost_version=xgb.__version__)
    booster.save_model(args.model_json)

    print(f"Saved model JSON to: {args.model_json}")
    print(f"#features: {len(feature_names)}")

    # ====== 可选：校准（仅当下游预测脚本会读取 attributes 才有效） ======
    # 若 attack_Di.py 直接 booster.predict 概率，这里属性不会被用到；但写上不影响兼容性
    if args.calib in ("temp", "platt"):
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        margins = booster.predict(dval, output_margin=True)

        if args.calib == "temp":
            bestT, bestNLL = 1.0, 1e9
            for T in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
                p = 1.0 / (1.0 + np.exp(-(margins / max(T, 1e-6))))
                nllT = log_loss(y_val, p, labels=[0, 1])
                if nllT < bestNLL:
                    bestNLL, bestT = nllT, T
            booster.set_attr(temp_scaling_T=str(bestT))
            print(f"[Calib] temperature T = {bestT:.3f} (val logloss={bestNLL:.6f})")

        elif args.calib == "platt":
            # 拟合 sigmoid(a*m + b)
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(margins.reshape(-1, 1), y_val)
            a = float(lr.coef_.ravel()[0]); b = float(lr.intercept_.ravel()[0])
            p = 1.0 / (1.0 + np.exp(-(a * margins + b)))
            nllP = log_loss(y_val, p, labels=[0, 1])
            booster.set_attr(platt_a=str(a))
            booster.set_attr(platt_b=str(b))
            print(f"[Calib] platt a={a:.6f}, b={b:.6f} (val logloss={nllP:.6f})")

    # 保存必要 attributes（不后编辑 JSON）
    booster.set_attr(feature_names=json.dumps(feature_names, ensure_ascii=False))
    booster.set_attr(target=args.target)
    booster.set_attr(xgboost_version=xgb.__version__)

    booster.save_model(args.model_json)
    print(f"Saved model JSON to: {args.model_json}")
    print(f"#features: {len(feature_names)}")


if __name__ == "__main__":
    main()
