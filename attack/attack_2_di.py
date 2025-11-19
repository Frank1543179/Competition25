# attack/attack_2_di.py
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
import sys
from typing import Tuple
CUR = Path(__file__).resolve().parent
ROOT = CUR.parent
sys.path.insert(0, str(ROOT / "analysis"))
from xgbt_train import build_X

TARGET = "stroke_flag"
NUM_ROWS = 100_000

def _loss_topk_by_class(D_path: Path, A_path: Path, topk=10_000, quota_mode="pred_ratio") -> np.ndarray:
    A = pd.read_csv(A_path, dtype=str, keep_default_na=False)
    y = pd.to_numeric(A[TARGET], errors="coerce").fillna(0).astype(int).values
    X = build_X(A, TARGET)

    bst = xgb.Booster(); bst.load_model(str(D_path))
    only_X = set(X.columns) - set(bst.feature_names)
    if only_X: X = X.drop(columns=only_X)
    for col in (set(bst.feature_names) - set(X.columns)): X[col]=0
    X = X.reindex(columns=bst.feature_names)

    p = bst.predict(xgb.DMatrix(X))
    p = np.clip(p, 1e-12, 1-1e-12)
    nll = -(y*np.log(p)+(1-y)*np.log(1-p))

    # 计算配额
    if quota_mode=="true_ratio":
        r1 = (y==1).mean()
    else:  # 用模型预测占比作为类比例（更稳）
        r1 = (p>=0.5).mean()
    q1 = int(round(topk * r1)); q0 = topk - q1

    idx1 = np.where(y==1)[0]; idx0 = np.where(y==0)[0]
    pick = []

    if len(idx1)>0 and q1>0:
        pick.extend(idx1[np.argsort(nll[idx1])[:min(q1, len(idx1))]])
    if len(idx0)>0 and q0>0:
        pick.extend(idx0[np.argsort(nll[idx0])[:min(q0, len(idx0))]])

    marks = np.zeros(len(A), dtype=int)
    marks[np.array(pick, dtype=int)] = 1
    # 若不足 topk，用整体最小 NLL 补齐
    if marks.sum() < topk:
        extra = np.argsort(nll)
        for i in extra:
            if marks[i]==0:
                marks[i]=1
                if marks.sum()>=topk: break
    return marks
