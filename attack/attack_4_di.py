# attack/attack_4_di.py
import argparse, sys
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

CUR = Path(__file__).resolve().parent
ROOT = CUR.parent
DATA = ROOT / "attack_data"
OUT = ROOT / "outputs4"

sys.path.insert(0, str(ROOT / "analysis"))
from xgbt_train import build_X

TARGET = "stroke_flag"

def _align(X: pd.DataFrame, booster: xgb.Booster) -> pd.DataFrame:
    need = set(booster.feature_names or [])
    have = set(X.columns)
    # 丢掉多余的，补缺失的
    X = X.drop(columns=(have - need), errors="ignore")
    for c in (need - set(X.columns)):
        X[c] = 0
    return X.reindex(columns=list(need))

def _leaf_idf_score(booster: xgb.Booster, X_ref: pd.DataFrame, X_tgt: pd.DataFrame) -> np.ndarray:
    dref = xgb.DMatrix(X_ref); dtgt = xgb.DMatrix(X_tgt)
    try:
        leaf_ref = booster.predict(dref, pred_leaf=True)
        leaf_tgt = booster.predict(dtgt, pred_leaf=True)
    except Exception:
        leaf_ref = booster.apply(dref)
        leaf_tgt = booster.apply(dtgt)
    T = leaf_tgt.shape[1]
    # 统计参考集（优先用 Ci）每棵树的叶子频率
    idf_tables = []
    for t in range(T):
        vals, cnts = np.unique(leaf_ref[:, t], return_counts=True)
        freq = cnts / cnts.sum()
        idf = {v: np.log(1.0 / (f + 1e-6)) for v, f in zip(vals, freq)}
        idf_tables.append(idf)
    # 目标样本在各树叶子的 IDF 累加
    s = np.zeros(leaf_tgt.shape[0], dtype=np.float64)
    for t in range(T):
        get = np.vectorize(lambda v: idf_tables[t].get(v, np.log(1.0/1e-6)))
        s += get(leaf_tgt[:, t])
    # 归一化
    return (s - s.min()) / (s.max() - s.min() + 1e-12)

def di_loss_topk(D_path: Path, A_path: Path, out_path: Path, topk=10_000, C_path: Path=None, lambda_leaf=0.6):
    Ai = pd.read_csv(A_path, dtype=str, keep_default_na=False)
    y = pd.to_numeric(Ai[TARGET], errors="coerce").fillna(0).astype(int).values
    XA = build_X(Ai, TARGET)

    booster = xgb.Booster(); booster.load_model(str(D_path))
    XA = _align(XA, booster); dA = xgb.DMatrix(XA)

    # 过拟合损失
    p = booster.predict(dA)
    p = np.clip(p, 1e-12, 1-1e-12)
    nll = -(y*np.log(p) + (1-y)*np.log(1-p))
    s_loss = (nll.max()-nll)/(nll.max()-nll.min()+1e-12)

    # 叶子 IDF：优先用 Ci（若有）；否则用 A 自身近似
    if C_path is not None and Path(C_path).exists():
        Ci = pd.read_csv(C_path, dtype=str, keep_default_na=False)
        XC = build_X(Ci if TARGET in Ci.columns else Ci.assign(**{TARGET:0}), TARGET)
        XC = _align(XC, booster)
        s_leaf = _leaf_idf_score(booster, XC, XA)
    else:
        s_leaf = _leaf_idf_score(booster, XA, XA)

    s = (1.0 - lambda_leaf) * s_loss + lambda_leaf * s_leaf
    order = np.argsort(-s)[:topk]
    marks = np.zeros(len(Ai), dtype=int); marks[order] = 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(marks, dtype=int).to_csv(out_path, index=False, header=False)
    print(f"[DI*] saved {out_path}  ones={int(marks.sum())}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("team", help="两位队号，如 07")
    ap.add_argument("--topk", type=int, default=10_000)
    ap.add_argument("--lambda_leaf", type=float, default=0.6)
    args = ap.parse_args()
    team = args.team.zfill(2)
    D = DATA / f"DD{team}.json"
    A = DATA / f"AA{team}.csv"
    C = DATA / f"CC{team}.csv"
    O = OUT / f"F{team}_DiLoss.csv"
    if not D.exists() or not A.exists():
        raise FileNotFoundError(f"缺少 {D} 或 {A}")
    di_loss_topk(D, A, O, topk=args.topk, C_path=(C if C.exists() else None),
                 lambda_leaf=args.lambda_leaf)
