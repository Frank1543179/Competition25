# attack/attack_2.py
import argparse, sys
import numpy as np
import pandas as pd
from pathlib import Path

# 路径
CUR  = Path(__file__).resolve().parent
ROOT = CUR.parent
DATA = ROOT / "attack_data"
OUT  = ROOT / "outputs2"

# 允许脚本式导入同目录模块
sys.path.insert(0, str(CUR))
from attack_2_ci import block_assign           # 你已改名的 CI（blocking + 一对一）
from attack_2_di import _loss_topk_by_class    # 你已改名的 DI（NLL+类配额）

NUM_ROWS = 100_000
TARGET_ONES = 10_000

def _exact_key_match(A: pd.DataFrame, C: pd.DataFrame, max_cat=10):
    """自动找一个稳定类别列做精确键匹配，返回命中的 Ai 索引集合（少量高精度）。"""
    common = [c for c in A.columns if c in C.columns]
    if not common: return set()

    def clean(df, cols):
        out = {}
        for c in cols:
            out[c] = df[c].astype(str).str.strip().str.lower().replace({"": "__miss__"})
        return pd.DataFrame(out)

    cat_cands = []
    for c in common:
        sA = A[c].astype(str).str.strip()
        frac_num = pd.to_numeric(sA.replace("", np.nan), errors="coerce").notna().mean()
        if frac_num < 0.5:
            cat_cands.append(c)
    if not cat_cands: return set()

    best, best_overlap = None, -1
    for c in cat_cands:
        vA = clean(A, [c])[c].value_counts()
        vC = clean(C, [c])[c].value_counts()
        overlap = list((set(vA.index) & set(vC.index)) - {"__miss__"})
        if 1 < len(overlap) <= max_cat and len(overlap) > best_overlap:
            best, best_overlap = (c,), len(overlap)
    if best is None: return set()

    Ak = clean(A, list(best)).agg("|".join, axis=1)
    Ck = clean(C, list(best)).agg("|".join, axis=1)

    hit = set()
    vcA = Ak.value_counts(); vcC = Ck.value_counts()
    keys = list(set(vcA.index) & set(vcC.index))
    rng = np.random.default_rng(2025)
    for key in keys:
        ia = np.where(Ak.values == key)[0]
        ic = np.where(Ck.values == key)[0]
        quota = min(len(ia), len(ic), 50)  # 单 key 最多取 50，避免爆表
        if quota <= 0: continue
        sel = rng.choice(ia, size=quota, replace=False)
        hit.update(map(int, sel))
    return hit

def _hard_hybrid_vector(team: str) -> np.ndarray:
    """返回长度=100000 的 0/1 向量（硬混合流水线）。"""
    A_path = DATA / f"AA{team}.csv"
    C_path = DATA / f"CC{team}.csv"
    D_path = DATA / f"DD{team}.json"

    have_ci = A_path.exists() and C_path.exists()
    have_di = A_path.exists() and D_path.exists()

    # 两边都没有，直接全0
    if not have_ci and not have_di:
        print(f"[HARD] team {team}: missing A/C AND A/D -> ZERO column")
        return np.zeros(NUM_ROWS, dtype=int)

    chosen = np.zeros(NUM_ROWS, dtype=int)

    # 1) 精确键（小量高精度）
    if have_ci:
        A = pd.read_csv(A_path, dtype=str, keep_default_na=False)
        C = pd.read_csv(C_path, dtype=str, keep_default_na=False)
        hit = _exact_key_match(A, C, max_cat=10)
        if hit:
            chosen[list(hit)] = 1

    # 2) Blocking + 一对一（写临时再合并）
    if have_ci:
        tmp_block = OUT / f"_tmp_block_{team}.csv"
        block_assign(A_path, C_path, tmp_block, k=7, metric="manhattan")  # 会写 10k 个 1
        bs = pd.read_csv(tmp_block, header=None).iloc[:, 0].astype(int).values
        bi = np.where(bs == 1)[0]
        need = TARGET_ONES - int(chosen.sum())
        for idx in bi:
            if need == 0: break
            if chosen[idx] == 0:
                chosen[idx] = 1; need -= 1

    # 3) DI-Loss 按类配额补齐
    if chosen.sum() < TARGET_ONES and have_di:
        di = _loss_topk_by_class(D_path, A_path, topk=TARGET_ONES, quota_mode="pred_ratio")
        di_idx = np.where(di == 1)[0]
        need = TARGET_ONES - int(chosen.sum())
        for idx in di_idx:
            if need == 0: break
            if chosen[idx] == 0:
                chosen[idx] = 1; need -= 1

    # 4) 兜底随机补齐
    if chosen.sum() < TARGET_ONES:
        pool = np.where(chosen == 0)[0]
        extra = np.random.default_rng(2025 + int(team)).choice(
            pool, size=TARGET_ONES - int(chosen.sum()), replace=False
        )
        chosen[extra] = 1

    return chosen

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("my", help="你的队号，例如 07")
    ap.add_argument("--own_zero", action="store_true", help="你自己的列全0（默认是空字符串）")
    ap.add_argument("--force_zero", type=str, default="", help="逗号分隔队号列表，强制这些列全0，如: 07,21")
    ap.add_argument("--overwrite", "-o", action="store_true")
    args = ap.parse_args()

    my = args.my.zfill(2)
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"F{my}.csv"
    if out_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{out_csv} 已存在，若要覆盖请加 -o/--overwrite")

    # 解析 force_zero
    force_zero = set([x.zfill(2) for x in args.force_zero.split(",") if x.strip()])

    # 构建 24 列
    cols = [f"{j:02d}" for j in range(1, 25)]
    Fi = pd.DataFrame(index=range(NUM_ROWS), columns=cols)

    for j in range(1, 25):
        team = f"{j:02d}"
        print(f"[HARD] building column {team} ...")

        # 强制全0优先
        if team in force_zero:
            Fi[team] = 0
            print(f"[HARD] team {team} forced to ZERO by --force_zero")
            continue

        # 你自己的列
        if team == my:
            Fi[team] = 0 if args.own_zero else ""   # 默认空字符串
            continue

        # 其他列：硬混合向量
        try:
            col = _hard_hybrid_vector(team)
        except Exception as e:
            print(f"[HARD] team {team} failed: {e} -> fallback random 10k")
            rng = np.random.default_rng(2025 + j)
            col = np.zeros(NUM_ROWS, dtype=int)
            col[rng.choice(NUM_ROWS, size=TARGET_ONES, replace=False)] = 1
        Fi[team] = col

    # 落盘（无表头/索引；NaN 写为空）
    Fi.to_csv(out_csv, index=False, header=False, na_rep="")

    # 写 id.txt 与 zip（可选）
    with open(OUT / "id.txt", "w", encoding="utf-8") as f:
        f.write(my)
    zip_path = OUT / f"F{my}.zip"
    from zipfile import ZipFile, ZIP_DEFLATED
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname=out_csv.name)
        zf.write(OUT / "id.txt", arcname="id.txt")

    print(f"[HARD] saved {out_csv} & {zip_path}")
