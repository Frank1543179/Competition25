# attack/attack_3.py  —— 只展示需要替换/新增的关键部分
import argparse, sys, numpy as np, pandas as pd
from pathlib import Path

CUR = Path(__file__).resolve().parent
ROOT = CUR.parent
DATA = ROOT / "attack_data"
OUT  = ROOT / "outputs3"

sys.path.insert(0, str(CUR))
from attack_3_ci import ci_vote_scores, ci_assign_plus   # 新
from attack_3_di import di_enhanced_scores               # 新

NUM_ROWS = 100_000
TARGET_ONES = 10_000

def load_col_csv(path: Path) -> np.ndarray:
    s = pd.read_csv(path, header=None).iloc[:,0]
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    s = (s > 0.5).astype(int).values
    if len(s) != NUM_ROWS:
        raise ValueError(f"{path} rows={len(s)} != {NUM_ROWS}")
    return s

def hybrid_make_col(team: str,
                    w_ci: float=0.6, w_di: float=0.4,
                    ci_rounds: int=40, ci_subspace: float=0.65, ci_k: int=5,
                    di_lambda_leaf: float=0.6) -> np.ndarray:
    A = DATA / f"AA{team}.csv"
    C = DATA / f"CC{team}.csv"
    D = DATA / f"DD{team}.json"

    have_ci = A.exists() and C.exists()
    have_di = A.exists() and D.exists()
    if not have_ci and not have_di:
        print(f"[HYB+] team {team}: missing A/C and A/D -> zeros")
        return np.zeros(NUM_ROWS, dtype=int)

    # 计算分数（缺哪条链就哪条为 0）
    s_ci = np.zeros(NUM_ROWS, dtype=np.float32)
    s_di = np.zeros(NUM_ROWS, dtype=np.float32)

    if have_ci:
        print(f"[HYB+] CI score ...")
        s_ci = ci_vote_scores(A, C, rounds=ci_rounds, subspace_frac=ci_subspace, k=ci_k, use_cosine=True, random_state=2025+int(team))
    else:
        print(f"[HYB+] no A/C")

    if have_di:
        print(f"[HYB+] DI score ...")
        # 若有 C，就把 C 传给 DI，启用叶子相似度对齐
        s_di = di_enhanced_scores(D, A, C_path=(C if C.exists() else None), lambda_leaf=di_lambda_leaf)
    else:
        print(f"[HYB+] no A/D")

    # 分数融合
    total = w_ci * s_ci + w_di * s_di
    if total.max()>0:
        total=(total-total.min())/(total.max()-total.min()+1e-12)

    order=np.argsort(-total)[:TARGET_ONES]
    chosen=np.zeros(NUM_ROWS, dtype=int)
    chosen[order]=1

    # 极端不足（几乎不可能发生）再随机补齐
    if chosen.sum() < TARGET_ONES:
        need = TARGET_ONES - int(chosen.sum())
        rng = np.random.default_rng(2025 + int(team))
        pool = np.where(chosen == 0)[0]
        extra = rng.choice(pool, size=need, replace=False)
        chosen[extra] = 1

    return chosen

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("my", help="你的队号，例如 07")
    ap.add_argument("--own_empty", action="store_true", help="你自己的那一列用空字符串（默认开启）")
    ap.add_argument("--own_zero", action="store_true", help="你自己的那一列用全0（如果评测要求全数字）")
    ap.add_argument("--overwrite", "-o", action="store_true")
    ap.add_argument("--force_zero", type=str, default="", help="逗号分隔队号列表，强制这些列全为0，例如: 07,21")
    # 可调但通常不需要改
    ap.add_argument("--w_ci", type=float, default=0.6)
    ap.add_argument("--w_di", type=float, default=0.4)
    ap.add_argument("--ci_rounds", type=int, default=40)
    ap.add_argument("--ci_subspace", type=float, default=0.65)
    ap.add_argument("--ci_k", type=int, default=5)
    ap.add_argument("--di_lambda_leaf", type=float, default=0.6)
    args = ap.parse_args()

    my = args.my.zfill(2)
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"F{my}.csv"
    if out_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{out_csv} 已存在，若要覆盖请加 -o")

    cols = [f"{j:02d}" for j in range(1, 25)]
    Fi = pd.DataFrame(index=range(NUM_ROWS), columns=cols)

    force_zero = set([x.zfill(2) for x in args.force_zero.split(",") if x.strip()])

    for j in range(1, 25):
        team = f"{j:02d}"
        print(f"[HYB+] building column {team} ...")
        if team in force_zero:
            Fi[team] = 0
            print(f"[HYB+] team {team} forced to ZERO by --force_zero")
            continue
        if team == my:
            Fi[team] = 0 if args.own_zero else ""
            continue

        try:
            col = hybrid_make_col(team,
                                  w_ci=args.w_ci, w_di=args.w_di,
                                  ci_rounds=args.ci_rounds, ci_subspace=args.ci_subspace, ci_k=args.ci_k,
                                  di_lambda_leaf=args.di_lambda_leaf)
        except Exception as e:
            print(f"[HYB+] team {team} failed: {e} -> fallback random 10k")
            rng = np.random.default_rng(2025 + j)
            col = np.zeros(NUM_ROWS, dtype=int)
            col[rng.choice(NUM_ROWS, size=TARGET_ONES, replace=False)] = 1
        Fi[team] = col

    Fi.to_csv(out_csv, index=False, header=False, na_rep="")
    from zipfile import ZipFile, ZIP_DEFLATED
    with open(OUT / "id.txt", "w", encoding="utf-8") as f: f.write(my)
    with ZipFile(OUT / f"F{my}.zip", "w", compression=ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname=out_csv.name); zf.write(OUT / "id.txt", arcname="id.txt")
    print(f"[HYB+] saved {out_csv} & {OUT / f'F{my}.zip'}")
