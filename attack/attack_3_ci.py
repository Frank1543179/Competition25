# attack/attack_3_ci.py
import numpy as np, pandas as pd
from pathlib import Path
from typing import Tuple, List
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances

def _read_str_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False)

def _infer_numeric(df: pd.DataFrame, cols: List[str], thresh=0.95) -> pd.Series:
    out=[]
    for c in cols:
        s=df[c].astype(str).str.strip()
        nonblank=s!=""
        if nonblank.sum()==0: out.append(False); continue
        conv=pd.to_numeric(s[nonblank], errors="coerce")
        out.append(conv.notna().mean()>=thresh)
    return pd.Series(out, index=cols)

def _prep_features(A: pd.DataFrame, C: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame,List[str],List[bool]]:
    common=[c for c in A.columns if c in C.columns]
    if not common:
        return (pd.DataFrame(index=A.index), pd.DataFrame(index=C.index), [], [])
    num_mask=_infer_numeric(A, common)
    num_cols=[c for c in common if num_mask[c]]
    cat_cols=[c for c in common if not num_mask[c]]

    # 数值：用 A 的均值/方差做 zscore，更稳（对极值再 winsor）
    X1n=pd.DataFrame(index=A.index); X2n=pd.DataFrame(index=C.index)
    if num_cols:
        A2=A[num_cols].replace({"":np.nan})
        C2=C[num_cols].replace({"":np.nan})
        A2=A2.apply(pd.to_numeric, errors="coerce"); C2=C2.apply(pd.to_numeric, errors="coerce")
        A2=A2.clip(A2.quantile(0.01), A2.quantile(0.99), axis=1)
        scaler=StandardScaler(with_mean=True, with_std=True)
        Afit=scaler.fit_transform(A2.fillna(A2.median()))
        Cfit=scaler.transform(C2.fillna(A2.median()))
        X1n=pd.DataFrame(Afit, index=A.index, columns=[f"num::{c}" for c in num_cols])
        X2n=pd.DataFrame(Cfit, index=C.index, columns=[f"num::{c}" for c in num_cols])

    # 类别：清洗 + 合并 one-hot
    X1c=pd.DataFrame(index=A.index); X2c=pd.DataFrame(index=C.index)
    if cat_cols:
        def clean(df):
            out=df.copy()
            for c in out.columns:
                s=out[c].astype(str).str.strip().str.lower().replace({"":"___missing___"})
                out[c]=s
            return out
        A3=clean(A[cat_cols]); C3=clean(C[cat_cols])
        both=pd.concat([A3,C3], axis=0, ignore_index=True)
        dummies=pd.get_dummies(both, dummy_na=False)
        nA=len(A3)
        X1c=dummies.iloc[:nA,:].set_index(A.index)
        X2c=dummies.iloc[nA:,:].set_index(C.index)
        X1c.columns=[f"cat::{c}" for c in X1c.columns]
        X2c.columns=X1c.columns

    X1=pd.concat([X1n,X1c], axis=1)
    X2=pd.concat([X2n,X2c], axis=1)

    feat_names=list(X1.columns)
    is_num=[name.startswith("num::") for name in feat_names]
    return X1, X2, feat_names, is_num

def _idf_weights_for_cats(X: pd.DataFrame, is_num: List[bool], eps=1e-6, power=1.2) -> np.ndarray:
    if X.shape[1]==0: return np.array([])
    freq=X.mean(axis=0).values  # 对 one-hot 等价于出现频率
    w=np.ones(X.shape[1], dtype=np.float32)
    for j, numflag in enumerate(is_num):
        if not numflag:
            w[j]=np.log(1.0/(freq[j]+eps))**power
    # 数值列做轻度标准化权重（信息量接近均衡）
    if any(is_num):
        num_idx=np.where(is_num)[0]
        # 以 1 为基准，避免把数值列打压太狠
        w[num_idx]=w[num_idx]*1.0
    return w

def _l2_normalize(X: np.ndarray, axis=1, eps=1e-12):
    nrm=np.linalg.norm(X, ord=2, axis=axis, keepdims=True)
    nrm=np.maximum(nrm, eps)
    return X/nrm

def ci_vote_scores(A_path: Path, C_path: Path,
                   rounds: int=40, subspace_frac: float=0.65,
                   k: int=5, use_cosine: bool=True, random_state: int=2025) -> np.ndarray:
    A=_read_str_csv(A_path); C=_read_str_csv(C_path)
    m=len(A)
    X1, X2, names, is_num=_prep_features(A,C)
    if X1.shape[1]==0 or X2.shape[1]==0 or m==0:
        return np.zeros(m, dtype=np.float32)

    W=_idf_weights_for_cats(pd.concat([X1,X2], axis=0, ignore_index=True), is_num)  # 依据整体频率
    X1w=X1.values*W
    X2w=X2.values*W

    rng=np.random.default_rng(random_state)
    scores=np.zeros(m, dtype=np.float32)

    for r in range(rounds):
        # 子空间采样
        d=X1w.shape[1]
        keep=max( min(d, int(np.ceil(d*subspace_frac))) , 8)
        cols=rng.choice(d, size=keep, replace=False)
        Xa=X1w[:,cols]
        Xb=X2w[:,cols]

        # 度量1：曼哈顿
        nn=NearestNeighbors(n_neighbors=min(k, max(1, len(A)//100)), metric="manhattan")
        nn.fit(Xa)
        dist, idx=nn.kneighbors(Xb, return_distance=True)
        w=1.0/(dist+1e-6)
        np.add.at(scores, idx.ravel(), w.ravel())

        # 度量2：余弦（对每轮可选）
        if use_cosine:
            Xa2=_l2_normalize(Xa, axis=1)
            Xb2=_l2_normalize(Xb, axis=1)
            # 直接余弦距离的 topk
            # 为避免 O(n*m)，用近邻再次求：先在 Xa2 上拟合欧氏近邻，余弦≈欧氏近似（已单位化）
            nn2=NearestNeighbors(n_neighbors=min(k, max(1, len(A)//100)), metric="euclidean")
            nn2.fit(Xa2)
            dist2, idx2=nn2.kneighbors(Xb2, return_distance=True)
            # 把欧氏距离映射为余弦相似度加权
            w2=1.0/(dist2+1e-6)
            np.add.at(scores, idx2.ravel(), w2.ravel())

    # 稀疏密度惩罚：被大量 Ci 命中的同一 Ai（热门簇）适度降权
    hit_counts=np.bincount(np.argmax(np.isfinite(scores).reshape(1,-1),axis=0), minlength=m)  # trick: 常量
    # 更可靠：soft 去趋势
    scores=scores/np.sqrt(1.0+0.002*hit_counts)

    # 归一化为 [0,1]
    if scores.max()>0:
        scores=(scores-scores.min())/(scores.max()-scores.min()+1e-12)
    return scores

def ci_assign_plus(A_path: Path, C_path: Path, out_marks: Path, target_ones=10_000, **kw):
    s=ci_vote_scores(A_path, C_path, **kw)
    order=np.argsort(-s)
    keep=order[:target_ones]
    marks=np.zeros(len(s), dtype=int); marks[keep]=1
    out_marks.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(marks).to_csv(out_marks, index=False, header=False)
    print(f"[CI+] saved {out_marks} ones={marks.sum()} (stability-vote)")
