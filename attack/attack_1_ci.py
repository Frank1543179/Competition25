# attack/attack_1_ci.py
import argparse, sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# 路径：.../project/attack/attack_1_ci.py
CUR = Path(__file__).resolve().parent
ROOT = CUR.parent
DATA = ROOT / "attack_data"
OUT = ROOT / "outputs1"

# 确保能 import mia.py（在 attack/ 里）
sys.path.insert(0, str(CUR))
from mia import build_feature_matrices  # 你现有的 mia.py

def ci_assign(A_path: Path, C_path: Path, out_path: Path, target_ones=10_000, k=7, metric="manhattan"):
    Ai_df = pd.read_csv(A_path, dtype=str, keep_default_na=False)
    Ci_df = pd.read_csv(C_path, dtype=str, keep_default_na=False)

    X1, X2 = build_feature_matrices(Ai_df, Ci_df)  # X1: Ai, X2: Ci
    m = len(Ai_df)
    if X1.shape[1] == 0 or X2.shape[1] == 0 or m == 0:
        pd.DataFrame(np.zeros(m, dtype=int)).to_csv(out_path, index=False, header=False)
        print(f"[CI] no common dims, saved zeros -> {out_path}")
        return

    k = max(1, min(k, X1.shape[0]))
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X1)
    dist, idx = nn.kneighbors(X2, n_neighbors=k, return_distance=True)

    pairs = []
    for ci_r in range(idx.shape[0]):
        for r in range(k):
            pairs.append((float(dist[ci_r, r]), r, int(idx[ci_r, r])))

    pairs.sort(key=lambda t: (t[0], t[1]))
    marks = np.zeros(m, dtype=int)
    used = set()
    picked = 0
    for d, rrank, ai_idx in pairs:
        if ai_idx in used:
            continue
        marks[ai_idx] = 1
        used.add(ai_idx)
        picked += 1
        if picked >= target_ones:
            break

    if picked < target_ones:
        # 用 Ai->Ci 最近距离补齐
        nn2 = NearestNeighbors(n_neighbors=1, metric=metric)
        nn2.fit(X2)
        dist_ai, _ = nn2.kneighbors(X1, n_neighbors=1, return_distance=True)
        order = np.argsort(dist_ai.ravel())
        for ai_idx in order:
            if ai_idx in used:
                continue
            marks[ai_idx] = 1
            used.add(ai_idx)
            picked += 1
            if picked >= target_ones:
                break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(marks, dtype=int).to_csv(out_path, index=False, header=False)
    print(f"[CI] saved {out_path}  ones={int(marks.sum())}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("team", help="两位队号，如 07")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--metric", type=str, default="manhattan")  # 可试 cosine
    args = ap.parse_args()

    team = args.team.zfill(2)
    A = DATA / f"AA{team}.csv"
    C = DATA / f"CC{team}.csv"
    O = OUT / f"F{team}_CiAssign.csv"

    if not A.exists() or not C.exists():
        raise FileNotFoundError(f"缺少 {A} 或 {C}")
    ci_assign(A, C, O, target_ones=10_000, k=args.k, metric=args.metric)
