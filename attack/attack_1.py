# attack/attack_1.py
import argparse, sys
import numpy as np
import pandas as pd
from pathlib import Path

CUR = Path(__file__).resolve().parent
ROOT = CUR.parent
DATA = ROOT / "attack_data"
OUT = ROOT / "outputs1"

# 引入刚才两个模块
sys.path.insert(0, str(CUR))
from attack_1_ci import ci_assign
from attack_1_di import di_loss_topk

NUM_ROWS = 100_000
TARGET_ONES = 10_000

def load_col_csv(path: Path) -> np.ndarray:
    s = pd.read_csv(path, header=None).iloc[:,0]
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    s = (s > 0.5).astype(int).values
    if len(s) != NUM_ROWS:
        raise ValueError(f"{path} rows={len(s)} != {NUM_ROWS}")
    return s

def hybrid_make_col(team: str) -> np.ndarray:
    A = DATA / f"AA{team}.csv"
    C = DATA / f"CC{team}.csv"
    D = DATA / f"DD{team}.json"
    tmp_ci = OUT / f"_tmp_CI_{team}.csv"
    tmp_di = OUT / f"_tmp_DI_{team}.csv"

    # ★ 新增：如果既没有 A/C，也没有 A/D，则直接返回全 0
    have_ci = A.exists() and C.exists()
    have_di = A.exists() and D.exists()
    if not have_ci and not have_di:
        print(f"[HYB] team {team}: missing A/C AND A/D -> force ZERO column")
        return np.zeros(NUM_ROWS, dtype=int)

    chosen = np.zeros(NUM_ROWS, dtype=int)

    # 1) CI（若有 A/C）
    if A.exists() and C.exists():
        ci_assign(A, C, tmp_ci, target_ones=TARGET_ONES, k=7, metric="manhattan")
        chosen = load_col_csv(tmp_ci)
    else:
        print(f"[HYB] team {team}: missing A/C, skip CI")

    # 2) DI（若有 A/D）
    if A.exists() and D.exists():
        di_loss_topk(D, A, tmp_di, topk=TARGET_ONES)
        di_marks = load_col_csv(tmp_di)
        # 从 DI 里按顺序补（用 DI 的 1 的索引顺序）
        di_idx = np.where(di_marks == 1)[0]
        need = TARGET_ONES - int(chosen.sum())
        for idx in di_idx:
            if need == 0:
                break
            if chosen[idx] == 0:
                chosen[idx] = 1
                need -= 1
    else:
        print(f"[HYB] team {team}: missing A/D, skip DI")

    # 3) 极端情况仍不足，随机补齐
    if chosen.sum() < TARGET_ONES:
        need = TARGET_ONES - int(chosen.sum())
        pool = np.where(chosen == 0)[0]
        rng = np.random.default_rng(2025 + int(team))
        extra = rng.choice(pool, size=need, replace=False)
        chosen[extra] = 1

    return chosen

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("my", help="你的队号，例如 07")
    ap.add_argument("--own_empty", action="store_true", help="你自己的那一列用空字符串（默认开启）")
    ap.add_argument("--own_zero", action="store_true", help="你自己的那一列用全0（如果评测要求全数字）")
    ap.add_argument("--overwrite", "-o", action="store_true")
    ap.add_argument("--force_zero", type=str, default="",help="逗号分隔队号列表，强制这些列全为0，例如: 07,21")

    args = ap.parse_args()

    my = args.my.zfill(2)
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"F{my}.csv"
    if out_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{out_csv} 已存在，若要覆盖请加 -o")

    # 建 24 列骨架
    cols = [f"{j:02d}" for j in range(1, 25)]
    Fi = pd.DataFrame(index=range(NUM_ROWS), columns=cols)

    # 解析参数后，构造一个集合
    force_zero = set([x.zfill(2) for x in args.force_zero.split(",") if x.strip()])

    for j in range(1, 25):
        team = f"{j:02d}"
        print(f"[HYB] building column {team} ...")
        if team in force_zero:
            Fi[team] = 0
            print(f"[HYB] team {team} forced to ZERO by --force_zero")
            continue
        if team == my:
            if args.own_zero:
                Fi[team] = 0
            else:
                # 默认空字符串（更通用）
                Fi[team] = ""
            continue

        try:
            col = hybrid_make_col(team)
        except Exception as e:
            print(f"[HYB] team {team} failed: {e}  -> fallback random 10k")
            rng = np.random.default_rng(2025 + j)
            col = np.zeros(NUM_ROWS, dtype=int)
            col[rng.choice(NUM_ROWS, size=TARGET_ONES, replace=False)] = 1

        Fi[team] = col

    # 落盘（无 header/index；把任何意外 NaN 写为空）
    Fi.to_csv(out_csv, index=False, header=False, na_rep="")
    # 生成 id.txt 和 zip（可选）
    with open(OUT / "id.txt", "w", encoding="utf-8") as f:
        f.write(my)
    zip_path = OUT / f"F{my}.zip"
    from zipfile import ZipFile, ZIP_DEFLATED
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname=out_csv.name)
        zf.write(OUT / "id.txt", arcname="id.txt")
    print(f"[HYB] saved {out_csv} & {zip_path}")
