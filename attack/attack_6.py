# attack/attack_fast.py
import argparse, sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

CUR = Path(__file__).resolve().parent
ROOT = CUR.parent
DATA = ROOT / "attack_data"
OUT = ROOT / "outputs_fast"

sys.path.insert(0, str(CUR))
try:
    from mia import build_feature_matrices
except ImportError:
    print("Warning: mia module not found")

NUM_ROWS = 100_000
TARGET_ONES = 10_000
TARGET = "stroke_flag"


class FastCIAttack:
    """快速CI攻击：减少轮数 + 优化特征处理"""

    def __init__(self, random_state: int = 2025):
        self.rng = np.random.default_rng(random_state)

    def fast_stability_voting(self, A_path: Path, C_path: Path,
                              n_rounds: int = 8,  # 大幅减少轮数
                              k_neighbors: int = 3) -> np.ndarray:  # 减少近邻数
        """快速稳定性投票"""
        A_df = pd.read_csv(A_path, dtype=str, keep_default_na=False)
        C_df = pd.read_csv(C_path, dtype=str, keep_default_na=False)

        # 使用原有的build_feature_matrices避免特征不匹配
        try:
            X1, X2 = build_feature_matrices(A_df, C_df)
            if X1.shape[1] == 0:
                return np.zeros(len(A_df), dtype=np.float32)
        except Exception as e:
            print(f"CI feature building failed: {e}, using fallback")
            return self._fallback_ci_attack(A_df, C_df)

        m = len(A_df)
        scores = np.zeros(m, dtype=np.float32)
        n_features = X1.shape[1]

        # 只使用曼哈顿距离，减少计算量
        from sklearn.neighbors import NearestNeighbors

        for round_idx in range(n_rounds):
            # 子空间采样（减少特征数）
            subspace_size = max(5, min(15, int(n_features * 0.6)))  # 限制特征数
            if n_features > subspace_size:
                features_idx = self.rng.choice(n_features, size=subspace_size, replace=False)
                X1_sub = X1[:, features_idx]
                X2_sub = X2[:, features_idx]
            else:
                X1_sub, X2_sub = X1, X2

            try:
                nn = NearestNeighbors(n_neighbors=k_neighbors, metric='manhattan', n_jobs=-1)  # 并行
                nn.fit(X1_sub)
                dist, idx = nn.kneighbors(X2_sub)

                # 快速权重计算
                weights = 1.0 / (dist + 1e-8)
                np.add.at(scores, idx.ravel(), weights.ravel())
            except Exception as e:
                print(f"CI round {round_idx} failed: {e}")
                continue

        # 快速归一化
        if scores.max() > scores.min() + 1e-8:
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores

    def _fallback_ci_attack(self, A_df: pd.DataFrame, C_df: pd.DataFrame) -> np.ndarray:
        """降级CI攻击：简单最近邻"""
        from sklearn.neighbors import NearestNeighbors

        try:
            # 简单数值化处理
            common_cols = [c for c in A_df.columns if c in C_df.columns and c != TARGET]
            if not common_cols:
                return np.zeros(len(A_df), dtype=np.float32)

            # 构建简单特征矩阵
            X1_list, X2_list = [], []
            for col in common_cols:
                try:
                    vals1 = pd.to_numeric(A_df[col], errors='coerce').fillna(0)
                    vals2 = pd.to_numeric(C_df[col], errors='coerce').fillna(0)
                    X1_list.append(vals1.values)
                    X2_list.append(vals2.values)
                except:
                    continue

            if not X1_list:
                return np.zeros(len(A_df), dtype=np.float32)

            X1 = np.column_stack(X1_list)
            X2 = np.column_stack(X2_list)

            # 单次最近邻搜索
            nn = NearestNeighbors(n_neighbors=1, metric='manhattan', n_jobs=-1)
            nn.fit(X1)
            idx = nn.kneighbors(X2, return_distance=False).ravel()

            scores = np.zeros(len(A_df), dtype=np.float32)
            scores[np.unique(idx)] = 1.0

            return scores

        except Exception as e:
            print(f"Fallback CI also failed: {e}")
            return np.zeros(len(A_df), dtype=np.float32)


class FastDIAttack:
    """快速DI攻击：简化特征计算"""

    def fast_di_scoring(self, D_path: Path, A_path: Path) -> np.ndarray:
        """快速DI评分：只使用关键特征"""
        try:
            import xgboost as xgb

            Ai = pd.read_csv(A_path, dtype=str, keep_default_na=False)
            y = pd.to_numeric(Ai[TARGET], errors='coerce').fillna(0).astype(int).values

            # 快速特征构建
            feature_cols = [c for c in Ai.columns if c != TARGET]
            X = Ai[feature_cols].copy()

            # 简单数值化
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

            # 加载模型
            booster = xgb.Booster()
            booster.load_model(str(D_path))

            # 快速特征对齐
            if hasattr(booster, 'feature_names') and booster.feature_names:
                for feat in booster.feature_names:
                    if feat not in X.columns:
                        X[feat] = 0
                X = X[booster.feature_names]

            dmatrix = xgb.DMatrix(X)

            # 只计算损失和置信度，跳过叶子相似度
            pred_proba = booster.predict(dmatrix)
            eps = 1e-12
            pred_proba = np.clip(pred_proba, eps, 1 - eps)

            # 1. 负对数似然损失（反转）
            nll = - (y * np.log(pred_proba) + (1 - y) * np.log(1 - pred_proba))
            s_loss = 1.0 - (nll - nll.min()) / (nll.max() - nll.min() + eps)

            # 2. 置信度评分
            confidence = np.where(y == 1, pred_proba, 1 - pred_proba)
            s_conf = (confidence - confidence.min()) / (confidence.max() - confidence.min() + eps)

            # 简单融合
            scores = 0.6 * s_loss + 0.4 * s_conf

            return scores

        except Exception as e:
            print(f"DI attack failed: {e}")
            return np.zeros(len(pd.read_csv(A_path)), dtype=np.float32)


class EfficientHybridAttack:
    """高效混合攻击"""

    def __init__(self):
        self.ci_attack = FastCIAttack()
        self.di_attack = FastDIAttack()

    def attack_team(self, team: str) -> np.ndarray:
        """攻击单个队伍"""
        A_path = DATA / f"AA{team}.csv"
        C_path = DATA / f"CC{team}.csv"
        D_path = DATA / f"DD{team}.json"

        have_ci = A_path.exists() and C_path.exists()
        have_di = A_path.exists() and D_path.exists()

        if not have_ci and not have_di:
            return np.zeros(NUM_ROWS, dtype=int)

        ci_scores, di_scores = np.zeros(NUM_ROWS), np.zeros(NUM_ROWS)

        # 并行计算CI和DI（在实际中可以改用多进程）
        if have_ci:
            try:
                ci_scores = self.ci_attack.fast_stability_voting(A_path, C_path)
                ci_available = True
            except Exception as e:
                print(f"Team {team} CI failed: {e}")
                ci_available = False
        else:
            ci_available = False

        if have_di:
            try:
                di_scores = self.di_attack.fast_di_scoring(D_path, A_path)
                di_available = True
            except Exception as e:
                print(f"Team {team} DI failed: {e}")
                di_available = False
        else:
            di_available = False

        # 动态权重
        if ci_available and di_available:
            # 简单权重：CI 60%, DI 40%
            combined_scores = 0.6 * ci_scores + 0.4 * di_scores
        elif ci_available:
            combined_scores = ci_scores
        elif di_available:
            combined_scores = di_scores
        else:
            # 降级到随机
            return self._random_fallback(team)

        # 选择top-k
        return self._select_top_k(combined_scores)

    def _select_top_k(self, scores: np.ndarray) -> np.ndarray:
        """选择top-k个样本"""
        if np.all(scores == 0):
            return self._random_fallback("fallback")

        # 使用argpartition提高性能（部分排序）
        if len(scores) > TARGET_ONES:
            indices = np.argpartition(scores, -TARGET_ONES)[-TARGET_ONES:]
        else:
            indices = np.arange(len(scores))

        result = np.zeros(len(scores), dtype=int)
        result[indices] = 1
        return result

    def _random_fallback(self, team: str) -> np.ndarray:
        """随机降级策略"""
        rng = np.random.default_rng(2025 + int(team) if team.isdigit() else 2025)
        result = np.zeros(NUM_ROWS, dtype=int)
        result[rng.choice(NUM_ROWS, size=TARGET_ONES, replace=False)] = 1
        return result


def main():
    ap = argparse.ArgumentParser(description="Fast Hybrid MIA Attack")
    ap.add_argument("my", help="Your team number, e.g., 07")
    ap.add_argument("--own_zero", action="store_true", help="Set your own column to zeros")
    ap.add_argument("--overwrite", "-o", action="store_true", help="Overwrite existing files")
    ap.add_argument("--force_zero", type=str, default="", help="Comma-separated team numbers to force zero")
    ap.add_argument("--skip_ci", action="store_true", help="Skip CI attacks to save time")
    ap.add_argument("--skip_di", action="store_true", help="Skip DI attacks to save time")

    args = ap.parse_args()

    my_team = args.my.zfill(2)
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"F{my_team}.csv"

    if out_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{out_csv} already exists. Use -o to overwrite.")

    print(f"[Fast] Starting attack for team {my_team}...")

    # 初始化攻击器
    attacker = EfficientHybridAttack()

    # 构建结果
    teams = [f"{i:02d}" for i in range(1, 25)]
    results = {}

    force_zero_teams = set([t.zfill(2) for t in args.force_zero.split(",") if t.strip()])

    for team in teams:
        if team in force_zero_teams:
            results[team] = np.zeros(NUM_ROWS, dtype=int)
            print(f"[Fast] Team {team} forced to zero")
            continue

        if team == my_team:
            results[team] = np.zeros(NUM_ROWS, dtype=int) if args.own_zero else np.array([""] * NUM_ROWS)
            continue

        # 根据参数跳过某些攻击
        if args.skip_ci and not (DATA / f"DD{team}.json").exists():
            print(f"[Fast] Skipping team {team} (no DI and CI skipped)")
            results[team] = attacker._random_fallback(team)
            continue

        if args.skip_di and not (DATA / f"CC{team}.csv").exists():
            print(f"[Fast] Skipping team {team} (no CI and DI skipped)")
            results[team] = attacker._random_fallback(team)
            continue

        print(f"[Fast] Attacking team {team}...")
        try:
            results[team] = attacker.attack_team(team)
            ones_count = np.sum(results[team])
            print(f"[Fast] Team {team} completed - ones: {ones_count}/10000")
        except Exception as e:
            print(f"[Fast] Team {team} failed: {e}, using random fallback")
            results[team] = attacker._random_fallback(team)

    # 构建DataFrame并保存
    result_df = pd.DataFrame(results)
    result_df.to_csv(out_csv, index=False, header=False, na_rep="")

    # 保存ID文件
    id_file = OUT / "id.txt"
    with open(id_file, "w") as f:
        f.write(my_team)

    print(f"[Fast] Successfully saved {out_csv}")
    print(f"[Fast] Total ones by column:")
    for team in teams:
        if team in results and hasattr(results[team], 'sum'):
            ones = results[team].sum() if isinstance(results[team], np.ndarray) else 0
            print(f"  Team {team}: {ones}")


if __name__ == "__main__":
    main()