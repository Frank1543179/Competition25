# attack/attack_2_block_assign.py
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import sys
CUR = Path(__file__).resolve().parent
sys.path.insert(0, str(CUR))
from mia import build_feature_matrices  # 你已有

NUM_ROWS = 100_000
TARGET_ONES = 10_000

def _pick_block_cols(A: pd.DataFrame, C: pd.DataFrame, max_cat=20):
    common = [c for c in A.columns if c in C.columns]
    if not common: return None, None
    # 识别类别列：非空值里 >90% 是非数字
    cat_cols, num_cols = [], []
    for c in common:
        s = A[c].astype(str).str.strip()
        frac_num = pd.to_numeric(s.replace("", np.nan), errors="coerce").notna().mean()
        (num_cols if frac_num >= 0.9 else cat_cols).append(c)
    # 选一个重叠度高且基数不大的类别列当 blockA
    best_cat = None; best_score = -1
    for c in cat_cols:
        vcA = A[c].astype(str).str.strip().str.lower().replace({"": "__miss__"}).value_counts()
        vcC = C[c].astype(str).str.strip().str.lower().replace({"": "__miss__"}).value_counts()
        overlap = set(vcA.index) & set(vcC.index)
        if 1 < len(overlap) <= max_cat:
            score = len(overlap) / max(len(vcA), 1)
            if score > best_score:
                best_score = score; best_cat = c
    # 选一个数值列用于分桶（可选）
    best_num = None
    for c in num_cols:
        x = pd.to_numeric(A[c].replace("", np.nan), errors="coerce")
        if x.notna().sum() > NUM_ROWS*0.3 and x.nunique() > 10:
            best_num = c; break
    return best_cat, best_num

def _bucket_num(s: pd.Series, q=5):
    x = pd.to_numeric(s.replace("", np.nan), errors="coerce")
    try:
        return pd.qcut(x, q=q, duplicates="drop").astype(str).fillna("__miss__")
    except Exception:
        return pd.Series(["__miss__"]*len(s))

def block_assign(A_path: Path, C_path: Path, out_path: Path, k=5, metric="manhattan"):
    A = pd.read_csv(A_path, dtype=str, keep_default_na=False)
    C = pd.read_csv(C_path, dtype=str, keep_default_na=False)
    m = len(A)
    marks = np.zeros(m, dtype=int)

    # 1) 选阻塞列
    blk_cat, blk_num = _pick_block_cols(A, C)
    if blk_cat is None:
        # 退化为全局匹配：调用你已有的一对一版本（或直接这里跑一次全局）
        from attack_1_ci import ci_assign  # 你已有
        ci_assign(A_path, C_path, out_path, target_ones=TARGET_ONES, k=7, metric=metric)
        return

    Ac = A[blk_cat].astype(str).str.strip().str.lower().replace({"": "__miss__"})
    Cc = C[blk_cat].astype(str).str.strip().str.lower().replace({"": "__miss__"})
    if blk_num:
        Ab = _bucket_num(A[blk_num], q=5)
        Cb = _bucket_num(C[blk_num], q=5)
        Akey = Ac + "|" + Ab
        Ckey = Cc + "|" + Cb
    else:
        Akey, Ckey = Ac, Cc

    # 2) 统计配额：按块大小近似分配 10k
    vcA = Akey.value_counts(); vcC = Ckey.value_counts()
    keys = list(set(vcA.index) & set(vcC.index))
    if not keys:
        from attack_1_ci import ci_assign
        ci_assign(A_path, C_path, out_path, target_ones=TARGET_ONES, k=7, metric=metric)
        return

    total = sum(min(vcA[k], vcC[k]) for k in keys)
    quotas = {k: int(round(TARGET_ONES * min(vcA[k], vcC[k]) / max(total, 1))) for k in keys}

    # 3) 每个块里做 kNN + 一对一分配
    used = set()
    left = TARGET_ONES
    for key in keys:
        if left <= 0: break
        idxA = np.where(Akey.values == key)[0]
        idxC = np.where(Ckey.values == key)[0]
        if len(idxA)==0 or len(idxC)==0: continue
        q = min(quotas[key], len(idxA))
        if q <= 0: continue

        X1, X2 = build_feature_matrices(A.iloc[idxA], C.iloc[idxC])
        if X1.shape[1]==0 or X2.shape[1]==0:
            continue
        k_eff = min(k, len(idxA))
        nn = NearestNeighbors(n_neighbors=k_eff, metric=metric).fit(X1)
        dist, nid = nn.kneighbors(X2, n_neighbors=k_eff, return_distance=True)

        pairs = []
        for r in range(nid.shape[0]):
            for t in range(k_eff):
                pairs.append((float(dist[r, t]), t, int(idxA[nid[r, t]])))
        pairs.sort(key=lambda x: (x[0], x[1]))

        picked = 0
        for d, rr, ai_idx in pairs:
            if ai_idx in used: continue
            marks[ai_idx]=1; used.add(ai_idx); picked+=1; left-=1
            if picked>=q or left<=0: break

    # 4) 全局补齐（用 A->C 最近距离）
    if marks.sum() < TARGET_ONES:
        X1, X2 = build_feature_matrices(A, C)
        if X1.shape[1]>0 and X2.shape[1]>0:
            nn2 = NearestNeighbors(n_neighbors=1, metric=metric).fit(X2)
            dA, _ = nn2.kneighbors(X1, n_neighbors=1, return_distance=True)
            order = np.argsort(dA.ravel())
            for ai in order:
                if marks[ai]==0:
                    marks[ai]=1
                    if marks.sum()>=TARGET_ONES: break

    pd.DataFrame(marks, dtype=int).to_csv(out_path, index=False, header=False)
    print(f"[BLOCK] {A_path.name} -> {out_path.name} ones={int(marks.sum())}")
