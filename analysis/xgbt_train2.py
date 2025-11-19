#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Privacy-oriented XGBoost trainer (JSON export, no post-edit)

- 前处理：数值化 + one-hot(drop_first=True) + 去零方差 + 列名升序固定
- 训练：原生 xgb.train（支持 soft label），强正则 + 子采样 + 以 logloss 早停
- 轻度隐私增强：
    * label noise（先少量随机翻转，再 label smoothing）
    * MixUp（可选）：X_mix = λ Xi + (1-λ) Xj,  y_mix = λ yi + (1-λ) yj
- 多随机种子训练，按验证集 logloss（次要：平均置信度）择优保存
- 保存前仅 set_attr，不后编辑 JSON
"""

import argparse, json, numpy as np, pandas as pd
from typing import List, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import xgboost as xgb

# ---------------- utils ----------------
def is_binary_01(s: pd.Series) -> bool:
    if s.empty: return False
    ss = s.astype(str).str.strip()
    if (ss == "").any(): return False
    num = pd.to_numeric(ss, errors="coerce")
    if num.isna().any(): return False
    vals = set(pd.unique(num.astype(int)))
    return vals.issubset({0,1}) and len(vals) > 0

def build_X(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        raise SystemExit(f"Target column '{target}' not found in CSV.")
    X_raw = df.drop(columns=[target]).copy()

    X_num_try = X_raw.apply(pd.to_numeric, errors="coerce")
    num_cols = [c for c in X_num_try.columns if X_num_try[c].notna().sum() > 0]
    X_num = X_num_try[num_cols] if num_cols else pd.DataFrame(index=df.index)

    cat_cols = [c for c in X_raw.columns if c not in num_cols]
    X_cat = pd.get_dummies(X_raw[cat_cols].astype("category"), drop_first=True) if cat_cols else pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_cat], axis=1)

    zero_var = X.nunique(dropna=False) <= 1
    if zero_var.all(): raise SystemExit("All feature columns have zero variance.")
    if zero_var.any(): X = X.loc[:, ~zero_var]

    X = X.reindex(columns=sorted(X.columns)).astype("float32")
    return X

def avg_confidence(proba: np.ndarray) -> float:
    """平均最大类置信度（越小越“谦虚”）"""
    return float(np.mean(np.maximum(proba, 1.0 - proba)))

# ------------- training core -------------
def train_one_seed(
    seed: int,
    X_tr: pd.DataFrame,
    y_tr_soft: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_names: List[str],
    args,
):
    # 组 DMatrix（soft label 在训练集）
    dtrain = xgb.DMatrix(X_tr.values, label=y_tr_soft, feature_names=feature_names)
    dval   = xgb.DMatrix(X_val.values, label=y_val,      feature_names=feature_names)

    # 类不平衡
    pos = float((y_val == 1).sum() + (y_tr_soft > 0.5).sum())  # 近似
    neg = float((y_val == 0).sum() + (y_tr_soft <= 0.5).sum())
    spw = (neg / pos) if pos > 0 else 1.0

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss","auc"],
        "tree_method": "hist",
        "eta": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "gamma": args.gamma,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "colsample_bylevel": args.colsample_bylevel,
        "max_delta_step": args.max_delta_step,
        "seed": seed,
        "nthread": -1,
        "scale_pos_weight": spw,
        # 更保守的分裂策略可选：
        # "grow_policy": "lossguide", "max_leaves": 32,
    }

    evals = [(dtrain, "train"), (dval, "val")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        evals=evals,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=False,
    )

    proba_val = booster.predict(dval)
    y_pred = (proba_val >= 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, proba_val)
    nll = log_loss(y_val, proba_val, labels=[0,1])
    conf = avg_confidence(proba_val)
    print(f"[seed {seed}] ACC={acc:.6f}  AUC={auc:.6f}  LOGLOSS={nll:.6f}  CONF={conf:.6f}  iters={booster.best_iteration+1}")
    return booster, nll, conf

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser(description="Privacy-oriented XGBoost trainer (JSON export).")
    ap.add_argument("train_csv", help="training CSV with header")
    ap.add_argument("--model-json", required=True, help="output model JSON path")
    ap.add_argument("--target", default="stroke_flag", help="binary target column (default: stroke_flag)")
    ap.add_argument("--test-size", type=float, default=0.1, help="validation ratio (default: 0.1)")

    # 更保守默认（可覆盖）
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--min-child-weight", type=float, default=10.0)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--reg-alpha", type=float, default=4.0)
    ap.add_argument("--reg-lambda", type=float, default=12.0)
    ap.add_argument("--subsample", type=float, default=0.6)
    ap.add_argument("--colsample-bytree", type=float, default=0.6)
    ap.add_argument("--colsample-bylevel", type=float, default=0.8)
    ap.add_argument("--max-delta-step", type=float, default=2.0)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--early-stopping-rounds", type=int, default=150)

    # 隐私增强
    ap.add_argument("--label-noise", type=float, default=0.015, help="randomly flip this fraction of train labels before smoothing")
    ap.add_argument("--label-smooth", type=float, default=0.03, help="epsilon in [0,0.1]")
    ap.add_argument("--mixup", action="store_true", help="enable MixUp on training set")
    ap.add_argument("--mixup-alpha", type=float, default=0.2, help="Beta(alpha,alpha)")
    ap.add_argument("--mixup-ratio", type=float, default=0.5, help="mixup samples count = ratio * n_train")

    # 多种子
    ap.add_argument("--seeds", default="7,13,29", help="comma-separated random seeds; best by val logloss is saved")
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv, dtype=str, keep_default_na=False)
    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' not found.")
    if not is_binary_01(df[args.target]):
        raise SystemExit(f"Target column '{args.target}' must be strictly binary 0/1.")

    y = pd.to_numeric(df[args.target], errors="coerce").astype(int).values
    X = build_X(df, target=args.target)
    feature_names: List[str] = list(X.columns)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=7, stratify=y
    )

    # ---------- Label noise + smoothing ----------
    rng = np.random.default_rng(2025)
    p_flip = max(0.0, min(0.25, float(args.label_noise)))
    if p_flip > 0:
        flip_mask = rng.random(len(y_tr)) < p_flip
        y_tr_noisy = y_tr.copy()
        y_tr_noisy[flip_mask] = 1 - y_tr_noisy[flip_mask]
    else:
        y_tr_noisy = y_tr

    eps = max(0.0, min(0.1, float(args.label_smooth)))
    y_tr_soft = y_tr_noisy * (1.0 - 2.0 * eps) + eps  # in [eps,1-eps]

    # ---------- MixUp (optional) ----------
    if args.mixup:
        n = X_tr.shape[0]
        n_mix = int(max(0, min(1.0, args.mixup_ratio)) * n)
        if n_mix > 0:
            i = rng.integers(0, n, size=n_mix)
            j = rng.integers(0, n, size=n_mix)
            lam = rng.beta(args.mixup_alpha, args.mixup_alpha, size=n_mix).astype("float32")
            Xi = X_tr.values[i]; Xj = X_tr.values[j]
            yi = y_tr_soft[i];   yj = y_tr_soft[j]
            X_mix = (lam[:, None] * Xi + (1.0 - lam)[:, None] * Xj).astype("float32")
            y_mix = (lam * yi + (1.0 - lam) * yj).astype("float32")

            X_tr = pd.DataFrame(
                np.vstack([X_tr.values, X_mix]),
                columns=feature_names
            )
            y_tr_soft = np.concatenate([y_tr_soft, y_mix], axis=0)

    # ---------- multi-seed training, pick best by val logloss ----------
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    best = {"booster": None, "nll": 1e9, "conf": 1e9, "seed": None}

    for sd in seeds:
        booster, nll, conf = train_one_seed(
            sd, X_tr, y_tr_soft, X_val, y_val, feature_names, args
        )
        if (nll < best["nll"]) or (abs(nll - best["nll"]) < 1e-6 and conf < best["conf"]):
            best.update({"booster": booster, "nll": nll, "conf": conf, "seed": sd})

    booster = best["booster"]
    print(f"[best] seed={best['seed']}  val logloss={best['nll']:.6f}  conf={best['conf']:.6f}")

    # 评估（再次）
    dval = xgb.DMatrix(X_val.values, label=y_val, feature_names=feature_names)
    proba_val = booster.predict(dval)
    y_pred = (proba_val >= 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, proba_val)
    nll = log_loss(y_val, proba_val, labels=[0,1])
    print(f"[best eval] ACC={acc:.6f}  AUC={auc:.6f}  LOGLOSS={nll:.6f}")

    # 写 attributes，保存 JSON（不后编辑）
    booster.set_attr(feature_names=json.dumps(feature_names, ensure_ascii=False))
    booster.set_attr(target=args.target)
    booster.set_attr(xgboost_version=xgb.__version__)
    booster.save_model(args.model_json)
    print(f"Saved model JSON to: {args.model_json}")
    print(f"#features: {len(feature_names)}")

if __name__ == "__main__":
    main()
