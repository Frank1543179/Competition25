# merge_24_to_one.py  (修改版)
import os
import numpy as np
import pandas as pd

DATA_DIR = "../attack_result1"
OUT_PATH = os.path.join(DATA_DIR, "F_all1.csv")
N_TEAMS = 24
N_ROWS = 100_000
ONES_PER_COL = 10_000

# 将“有问题的数据组”填到这里（两位字符串），第7组会自动加入
BAD_TEAMS = {"07","21"}  #07是自己队伍，21无数据

def load_col_or_raise(team_id: str) -> pd.Series:
    cand1 = os.path.join(DATA_DIR, f"F{team_id}_mix.csv")
    cand2 = os.path.join(DATA_DIR, f"F{team_id}_Ci.csv")
    path = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)
    if path is None:
        raise FileNotFoundError(f"找不到 F{team_id}_mix.csv 或 F{team_id}_Ci.csv（期望在 {DATA_DIR}）")
    s = pd.read_csv(path, header=None).iloc[:, 0]
    if len(s) != N_ROWS:
        raise ValueError(f"{path} 行数为 {len(s)}，应为 {N_ROWS}")
    # 归一成 0/1
    s = (pd.to_numeric(s, errors="coerce").fillna(0) > 0.5).astype(int)
    s.index = range(N_ROWS)
    s.name = team_id
    return s

def force_ones_count(s: pd.Series, team_id: str) -> pd.Series:
    """将该列强制调整为恰好 ONES_PER_COL 个 1（其余 0），可复现随机。"""
    rng = np.random.default_rng(2025 + int(team_id))
    vals = s.to_numpy().astype(int, copy=True)
    ones_idx = np.where(vals == 1)[0]
    zeros_idx = np.where(vals == 0)[0]

    if len(ones_idx) > ONES_PER_COL:
        keep = rng.choice(ones_idx, size=ONES_PER_COL, replace=False)
        new_vals = np.zeros(N_ROWS, dtype=int)
        new_vals[keep] = 1
        vals = new_vals
    elif len(ones_idx) < ONES_PER_COL:
        need = ONES_PER_COL - len(ones_idx)
        add = rng.choice(zeros_idx, size=need, replace=False)
        vals[add] = 1
    # 等于时不变

    return pd.Series(vals, name=team_id)

def build_matrix() -> pd.DataFrame:
    cols = []
    for j in range(1, N_TEAMS + 1):
        team_id = f"{j:02d}"

        # 第7组和“有问题的数据组”强制全0
        if team_id == "07" or team_id in BAD_TEAMS:
            s = pd.Series(np.zeros(N_ROWS, dtype=int), name=team_id)
            cols.append(s)
            continue

        # 正常加载
        s = load_col_or_raise(team_id)
        # 强制 1 的数量为 10,000
        s = force_ones_count(s, team_id)
        cols.append(s)

    df = pd.concat(cols, axis=1)
    return df

if __name__ == "__main__":
    # 确保第7组在 BAD_TEAMS 里（双保险）
    BAD_TEAMS.add("07")

    df = build_matrix()

    # 校验：每列都是 0/1，且行数对
    assert df.shape == (N_ROWS, N_TEAMS), f"形状异常：{df.shape}"
    assert set(np.unique(df.to_numpy())) <= {0, 1}, "存在非0/1值"

    # 校验：第7列全0
    assert df["07"].sum() == 0, "第7组不是全0"

    # 校验：非 BAD_TEAMS 的列恰好 10000 个 1
    for c in df.columns:
        if c in BAD_TEAMS:
            continue
        ones = int(df[c].sum())
        if ones != ONES_PER_COL:
            raise AssertionError(f"列 {c} 的 1 的数量为 {ones}，应为 {ONES_PER_COL}")

    # 保存：无表头、无索引
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, header=False, index=False)
    print(f"已保存：{OUT_PATH}  形状 {df.shape}")
    print("各列1的数量（前若干）：", df.sum(axis=0).head().to_dict())
