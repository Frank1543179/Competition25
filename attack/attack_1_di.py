# attack/attack_1_di.py
from abc import ABC
import argparse, sys
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

CUR = Path(__file__).resolve().parent
ROOT = CUR.parent
DATA = ROOT / "attack_data"
OUT = ROOT / "outputs1"

# 让脚本能 import analysis/xgbt_train.py
sys.path.insert(0, str(ROOT / "analysis"))
from xgbt_train import build_X  # 你现有的

TARGET = "stroke_flag"

def di_loss_topk(D_path: Path, A_path: Path, out_path: Path, topk=10_000):
    Ai = pd.read_csv(A_path, dtype=str, keep_default_na=False)
    y = pd.to_numeric(Ai[TARGET], errors="coerce").fillna(0).astype(int).values
    X = build_X(Ai, TARGET)

    booster = xgb.Booster()
    booster.load_model(str(D_path))

    # 对齐特征
    cols_only_X = set(X.columns) - set(booster.feature_names)
    if cols_only_X:
        X = X.drop(columns=cols_only_X)
    for col in (set(booster.feature_names) - set(X.columns)):
        X[col] = 0
    X = X.reindex(columns=booster.feature_names)

    p = booster.predict(xgb.DMatrix(X))
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    nll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    order = np.argsort(nll)
    keep = order[:topk]

    marks = np.zeros(len(Ai), dtype=int)
    marks[keep] = 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(marks, dtype=int).to_csv(out_path, index=False, header=False)
    print(f"[DI] saved {out_path}  ones={int(marks.sum())}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("team", help="两位队号，如 07")
    ap.add_argument("--topk", type=int, default=10_000)
    args = ap.parse_args()

    team = args.team.zfill(2)
    D = DATA / f"DD{team}.json"
    A = DATA / f"AA{team}.csv"
    O = OUT / f"F{team}_DiLoss.csv"

    if not D.exists() or not A.exists():
        raise FileNotFoundError(f"缺少 {D} 或 {A}")
    di_loss_topk(D, A, O, topk=args.topk)
