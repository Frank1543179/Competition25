# attack/attack_3_di.py
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
import sys
CUR=Path(__file__).resolve().parent
ROOT=CUR.parent
sys.path.insert(0, str(ROOT/"analysis"))
from xgbt_train import build_X  # 和你现有的一致

TARGET="stroke_flag"

def _align_features(X: pd.DataFrame, booster: xgb.Booster) -> pd.DataFrame:
    cols_only_X=set(X.columns)-set(booster.feature_names or [])
    if cols_only_X: X=X.drop(columns=cols_only_X)
    for col in (set(booster.feature_names or []) - set(X.columns)):
        X[col]=0
    return X.reindex(columns=booster.feature_names)

def di_enhanced_scores(D_path: Path, A_path: Path, C_path: Path=None, lambda_leaf: float=0.6) -> np.ndarray:
    Ai=pd.read_csv(A_path, dtype=str, keep_default_na=False)
    y=pd.to_numeric(Ai[TARGET], errors="coerce").fillna(0).astype(int).values
    XA=build_X(Ai, TARGET)

    booster=xgb.Booster(); booster.load_model(str(D_path))
    XA=_align_features(XA, booster)
    dA=xgb.DMatrix(XA)

    # 1) 过拟合损失（成员倾向：NLL 更小）
    p=booster.predict(dA)
    eps=1e-12; p=np.clip(p, eps, 1-eps)
    nll=-(y*np.log(p)+(1-y)*np.log(1-p))
    s_loss=(nll.max()-nll)/(nll.max()-nll.min()+1e-12)  # 小损失→大分数

    # 2) 叶子相似度（可选用 Ci 做分布刻画；无 Ci 时退化为“稀有叶子”打分）
    try:
        leafA=booster.predict(dA, pred_leaf=True)  # shape [m, n_trees]
    except Exception:
        leafA=booster.apply(dA)  # 有些版本用 apply
    m, T = leafA.shape[0], leafA.shape[1]

    if C_path is not None and Path(C_path).exists():
        Ci=pd.read_csv(C_path, dtype=str, keep_default_na=False)
        XC=build_X(Ci, TARGET) if TARGET in Ci.columns else build_X(Ci.assign(**{TARGET:0}), TARGET)
        XC=_align_features(XC, booster)
        dC=xgb.DMatrix(XC)
        try:
            leafC=booster.predict(dC, pred_leaf=True)
        except Exception:
            leafC=booster.apply(dC)

        # 统计每棵树 Ci 的叶子频率，IDF 权重：罕见叶子价值高
        # 为每个树 t 建立 dict: leaf_id -> freq
        freq_list=[]
        for t in range(T):
            vals, cnts=np.unique(leafC[:,t], return_counts=True)
            freq={v:c/len(leafC) for v,c in zip(vals, cnts)}
            freq_list.append(freq)
        sim=np.zeros(m, dtype=np.float32)
        for i in range(m):
            s=0.0
            for t in range(T):
                v=leafA[i,t]
                f=freq_list[t].get(v, 0.0)
                # 叶子出现在 Ci 且越罕见，信息量越大
                s += np.log(1.0/(f+1e-6))
            sim[i]=s
        s_leaf=(sim - sim.min())/(sim.max()-sim.min()+1e-12)
    else:
        # 无 Ci：用 A 自身的稀有叶子作为 proxy
        sim=np.zeros(m, dtype=np.float32)
        for t in range(T):
            vals, cnts=np.unique(leafA[:,t], return_counts=True)
            invf=dict((v, np.log(1.0/(c/m+1e-6))) for v,c in zip(vals,cnts))
            sim += np.vectorize(invf.get)(leafA[:,t])
        s_leaf=(sim - sim.min())/(sim.max()-sim.min()+1e-12)

    # 融合
    s = (1.0 - lambda_leaf) * s_loss + lambda_leaf * s_leaf
    if s.max()>0: s=(s - s.min())/(s.max()-s.min()+1e-12)
    return s

def di_loss_topk_plus(D_path: Path, A_path: Path, out_path: Path, C_path: Path=None, topk=10_000, lambda_leaf=0.6):
    s=di_enhanced_scores(D_path, A_path, C_path=C_path, lambda_leaf=lambda_leaf)
    order=np.argsort(-s); keep=order[:topk]
    marks=np.zeros(len(s), dtype=int); marks[keep]=1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(marks, dtype=int).to_csv(out_path, index=False, header=False)
    print(f"[DI+] saved {out_path} ones={marks.sum()} (loss+leaf-sim)")
