# attack/attack_4_ci.py
import argparse, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

CUR = Path(__file__).resolve().parent
ROOT = CUR.parent
DATA = ROOT / "attack_data"
OUT = ROOT / "outputs4"

sys.path.insert(0, str(CUR))
from mia import build_feature_matrices  # 保持你的 mia.py


# --------- 新增：找稳定低基数类别列做“精确键”分桶 ----------
def _infer_numeric_mask(df: pd.DataFrame, cols, thresh=0.95):
    mask = []
    for c in cols:
        s = df[c].astype(str).str.strip()
        nonblank = s != ""
        if nonblank.sum() == 0: mask.append(False); continue
        frac = pd.to_numeric(s[nonblank], errors="coerce").notna().mean()
        mask.append(frac >= thresh)
    return pd.Series(mask, index=cols)


def _stable_cats(A: pd.DataFrame, C: pd.DataFrame, max_cat=10, pick=2):
    common = [c for c in A.columns if c in C.columns]
    if not common: return []
    num_mask = _infer_numeric_mask(A, common)
    cand = [c for c in common if not num_mask[c]]
    # 低基数 + 两侧都有值
    stats = []
    for c in cand:
        uA = A[c].astype(str).str.strip().replace({"": np.nan}).dropna().nunique()
        uC = C[c].astype(str).str.strip().replace({"": np.nan}).dropna().nunique()
        if 1 <= uA <= max_cat and 1 <= uC <= max_cat:
            # 重叠度越大越好
            inter = len(set(A[c].astype(str)) & set(C[c].astype(str)))
            stats.append((-(uA + uC), inter, c))
    stats.sort()
    return [c for _, __, c in stats[:pick]]


def _make_block_keys(df: pd.DataFrame, cols):
    if not cols:
        return pd.Series(["__all__"] * len(df), index=df.index)
    out = []
    for c in cols:
        out.append(df[c].astype(str).str.strip().replace({"": "__miss__"}))
    key = pd.concat(out, axis=1).agg("\x1f".join, axis=1)  # 用不可见分隔
    return key


# --------- 主函数：分桶 + 多邻居投票（保留原函数签名） ----------
def ci_assign(A_path: Path, C_path: Path, out_path: Path, target_ones=10_000,
              k=7, metric="manhattan", rounds=1, block_max_cat=10, vote_power=1.0):
    Ai = pd.read_csv(A_path, dtype=str, keep_default_na=False)
    Ci = pd.read_csv(C_path, dtype=str, keep_default_na=False)

    # 先用 mia 的特征构建（保证与原版一致的预处理）
    X1, X2 = build_feature_matrices(Ai, Ci)  # numpy
    m = len(Ai)
    if X1.shape[1] == 0 or X2.shape[1] == 0 or m == 0:
        pd.DataFrame(np.zeros(m, dtype=int)).to_csv(out_path, index=False, header=False)
        print(f"[CI*] no common dims, saved zeros -> {out_path}")
        return

    # 稀有 one-hot 维度加权（用稀疏度近似 IDF；均一化到乘法缩放）
    feat_mean = X1.mean(axis=0)
    idf = np.log(1.0 / (feat_mean + 1e-6)) ** 1.2
    # 防止把数值维（密集）打压太多：截断到 [0.5, 3] 然后整体再 z 标准化到 ~1 附近
    idf = np.clip(idf, 0.5, 3.0)
    X1w = X1 * idf
    X2w = X2 * idf

    # —— 精确键分桶（低基数 1~2 列）：
    cat_cols = _stable_cats(Ai, Ci, max_cat=block_max_cat, pick=2)
    keyA = _make_block_keys(Ai, cat_cols)
    keyC = _make_block_keys(Ci, cat_cols)

    # 建索引：每个桶只在桶内做近邻，极大增加可比性与“峰值”
    buckets = {}
    for val in keyA.unique():
        idx = np.where(keyA.values == val)[0]
        buckets[val] = {"A_idx": idx, "X1": X1w[idx]}

    # 投票分数（按“Ci→Ai 多邻居 + 距离倒数^power”）
    votes = np.zeros(m, dtype=np.float64)

    # 允许多轮以降低偶然性（默认 rounds=1 和你原方法一样快）
    for _ in range(max(1, int(rounds))):
        for b in pd.unique(keyC):
            Ainfo = buckets.get(b)
            if Ainfo is None or len(Ainfo["A_idx"]) == 0:
                # 桶里没 A，用全体回退（很少发生）
                Xa = X1w;
                Aidx = np.arange(m)
            else:
                Xa = Ainfo["X1"];
                Aidx = Ainfo["A_idx"]

            # 取出本桶的 Ci
            qb = (keyC.values == b)
            Xq = X2w[qb]
            if Xq.shape[0] == 0:
                continue

            k_use = max(1, min(k, len(Aidx)))
            nn = NearestNeighbors(n_neighbors=k_use, metric=metric)
            nn.fit(Xa)
            dist, nbr = nn.kneighbors(Xq, n_neighbors=k_use, return_distance=True)

            # 距离越小票越重；对 rank 再除以 (rank+1)
            w = (1.0 / (dist + 1e-8)) ** vote_power
            ranks = np.arange(1, k_use + 1)[None, :]
            w = w / ranks
            # 汇总到全局 Ai 索引
            np.add.at(votes, Aidx[nbr.ravel()], w.ravel())

    # 把“桶内完全一致”的 A（即同时在 C 出现相同键的）再给一个固定偏置（放大峰值）
    if len(cat_cols) > 0:
        appear = pd.Series(0, index=np.arange(m))
        appear.loc[np.isin(keyA.values, keyC.unique())] = 1
        votes += appear.values * (votes.mean() + votes.std())  # 自适应 bonus

    # 选 Top-K（不再做“唯一分配”扩散，保持你想要的集中化峰值）
    order = np.argsort(-votes)
    keep = order[:target_ones]
    marks = np.zeros(m, dtype=int)
    marks[keep] = 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(marks, dtype=int).to_csv(out_path, index=False, header=False)
    print(f"[CI*] saved {out_path}  ones={int(marks.sum())}  keys={cat_cols if cat_cols else 'NONE'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("team", help="两位队号，如 07")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--metric", type=str, default="manhattan")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--block_max_cat", type=int, default=10)
    ap.add_argument("--vote_power", type=float, default=1.0)
    args = ap.parse_args()
    team = args.team.zfill(2)
    A = DATA / f"AA{team}.csv"
    C = DATA / f"CC{team}.csv"
    O = OUT / f"F{team}_CiAssign.csv"
    if not A.exists() or not C.exists():
        raise FileNotFoundError(f"缺少 {A} 或 {C}")
    ci_assign(A, C, O, target_ones=10_000, k=args.k, metric=args.metric,
              rounds=args.rounds, block_max_cat=args.block_max_cat, vote_power=args.vote_power)
