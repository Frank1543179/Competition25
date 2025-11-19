# attack/attack_enhanced.py
import argparse, sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

CUR = Path(__file__).resolve().parent
ROOT = CUR.parent
DATA = ROOT / "attack_data"
OUT = ROOT / "outputs_enhanced"

# 引入必要的模块
sys.path.insert(0, str(CUR))
try:
    from mia import build_feature_matrices
except ImportError:
    print("Warning: mia module not found, using fallback")

try:
    sys.path.insert(0, str(ROOT / "analysis"))
    from xgbt_train import build_X
except ImportError:
    print("Warning: xgbt_train module not found, using fallback")

NUM_ROWS = 100_000
TARGET_ONES = 10_000
TARGET = "stroke_flag"


class EnhancedCIAttack:
    """增强的CI攻击：结合稳定性投票和自适应特征加权"""

    def __init__(self, random_state: int = 2025):
        self.rng = np.random.default_rng(random_state)

    def _prepare_features(self, A_df: pd.DataFrame, C_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备特征矩阵，包含数值和类别特征的智能处理"""
        try:
            X1, X2 = build_feature_matrices(A_df, C_df)
            return X1.values, X2.values
        except:
            # Fallback: 简单数值化处理
            common_cols = [c for c in A_df.columns if c in C_df.columns and c != TARGET]
            if not common_cols:
                return np.empty((len(A_df), 0)), np.empty((len(C_df), 0))

            X1_list, X2_list = [], []
            for col in common_cols:
                try:
                    # 尝试数值化
                    vals1 = pd.to_numeric(A_df[col], errors='coerce').fillna(0).values
                    vals2 = pd.to_numeric(C_df[col], errors='coerce').fillna(0).values
                    # 标准化
                    mean, std = vals1.mean(), vals1.std() + 1e-8
                    X1_list.append((vals1 - mean) / std)
                    X2_list.append((vals2 - mean) / std)
                except:
                    # 类别特征: 简单哈希编码
                    all_vals = pd.concat([A_df[col], C_df[col]], ignore_index=True)
                    unique_vals = all_vals.unique()
                    val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
                    X1_list.append([val_to_idx.get(v, 0) for v in A_df[col]])
                    X2_list.append([val_to_idx.get(v, 0) for v in C_df[col]])

            return np.column_stack(X1_list), np.column_stack(X2_list)

    def stability_voting(self, A_path: Path, C_path: Path,
                         n_rounds: int = 30, subspace_ratio: float = 0.7,
                         k_neighbors: int = 5) -> np.ndarray:
        """稳定性投票：多轮子空间采样+多距离度量"""
        A_df = pd.read_csv(A_path, dtype=str, keep_default_na=False)
        C_df = pd.read_csv(C_path, dtype=str, keep_default_na=False)

        X1, X2 = self._prepare_features(A_df, C_df)
        m = len(A_df)

        if X1.shape[1] == 0 or m == 0:
            return np.zeros(m, dtype=np.float32)

        scores = np.zeros(m, dtype=np.float32)
        n_features = X1.shape[1]
        subspace_size = max(3, int(n_features * subspace_ratio))

        for round_idx in range(n_rounds):
            # 子空间采样
            if n_features > subspace_size:
                features_idx = self.rng.choice(n_features, size=subspace_size, replace=False)
                X1_sub = X1[:, features_idx]
                X2_sub = X2[:, features_idx]
            else:
                X1_sub, X2_sub = X1, X2

            # 多距离度量融合
            for metric in ['manhattan', 'cosine']:
                try:
                    from sklearn.neighbors import NearestNeighbors
                    from sklearn.metrics.pairwise import cosine_distances

                    if metric == 'cosine':
                        # 余弦距离
                        distances = cosine_distances(X2_sub, X1_sub)
                        for i in range(len(C_df)):
                            nearest_idx = np.argsort(distances[i])[:k_neighbors]
                            weights = 1.0 / (distances[i, nearest_idx] + 1e-8)
                            scores[nearest_idx] += weights
                    else:
                        # 曼哈顿距离
                        nn = NearestNeighbors(n_neighbors=k_neighbors, metric='manhattan')
                        nn.fit(X1_sub)
                        dist, idx = nn.kneighbors(X2_sub)
                        weights = 1.0 / (dist + 1e-8)
                        np.add.at(scores, idx.ravel(), weights.ravel())
                except:
                    continue

        # 热度惩罚：避免过度集中
        if scores.max() > 0:
            hit_density = np.zeros(m)
            for round_idx in range(min(5, n_rounds)):
                if n_features > 3:
                    features_idx = self.rng.choice(n_features, size=3, replace=False)
                    X1_sub = X1[:, features_idx]
                    nn = NearestNeighbors(n_neighbors=1, metric='manhattan')
                    nn.fit(X1_sub)
                    _, idx = nn.kneighbors(X2_sub)
                    np.add.at(hit_density, idx.ravel(), 1)

            penalty = 1.0 / (1.0 + 0.1 * hit_density)
            scores *= penalty

            # 归一化
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

        return scores


class EnhancedDIAttack:
    """增强的DI攻击：多特征融合 + 置信度校准"""

    def __init__(self):
        self.feature_importance = None

    def _align_features(self, X: pd.DataFrame, booster) -> pd.DataFrame:
        """对齐特征到模型期望的格式"""
        if hasattr(booster, 'feature_names') and booster.feature_names:
            model_features = set(booster.feature_names)
            current_features = set(X.columns)

            # 移除多余特征
            extra_features = current_features - model_features
            if extra_features:
                X = X.drop(columns=list(extra_features))

            # 添加缺失特征
            missing_features = model_features - current_features
            for feat in missing_features:
                X[feat] = 0

            # 重新排序
            X = X[booster.feature_names]

        return X

    def multi_feature_scoring(self, D_path: Path, A_path: Path,
                              C_path: Optional[Path] = None) -> np.ndarray:
        """多特征评分：损失、置信度、叶子相似度融合"""
        try:
            import xgboost as xgb

            Ai = pd.read_csv(A_path, dtype=str, keep_default_na=False)
            y = pd.to_numeric(Ai[TARGET], errors='coerce').fillna(0).astype(int).values

            # 构建特征
            try:
                X = build_X(Ai, TARGET)
            except:
                # Fallback: 简单数值化
                feature_cols = [c for c in Ai.columns if c != TARGET]
                X = Ai[feature_cols].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0))

            # 加载模型和对齐特征
            booster = xgb.Booster()
            booster.load_model(str(D_path))
            X = self._align_features(X, booster)
            dmatrix = xgb.DMatrix(X)

            # 1. 负对数似然损失
            pred_proba = booster.predict(dmatrix)
            eps = 1e-12
            pred_proba = np.clip(pred_proba, eps, 1 - eps)
            nll = - (y * np.log(pred_proba) + (1 - y) * np.log(1 - pred_proba))
            s_loss = 1.0 - (nll - nll.min()) / (nll.max() - nll.min() + eps)

            # 2. 置信度评分
            confidence = np.where(y == 1, pred_proba, 1 - pred_proba)
            s_conf = (confidence - confidence.min()) / (confidence.max() - confidence.min() + eps)

            # 3. 预测正确性
            pred_label = (pred_proba >= 0.5).astype(int)
            s_correct = (pred_label == y).astype(float)

            # 4. 叶子相似度（如果可用）
            s_leaf = np.zeros(len(Ai))
            try:
                leaf_pred = booster.predict(dmatrix, pred_leaf=True)
                if C_path and C_path.exists():
                    # 使用C计算叶子频率
                    Ci = pd.read_csv(C_path, dtype=str, keep_default_na=False)
                    try:
                        Xc = build_X(Ci, TARGET)
                    except:
                        feature_cols = [c for c in Ci.columns if c != TARGET]
                        Xc = Ci[feature_cols].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0))
                    Xc = self._align_features(Xc, booster)
                    dmatrix_c = xgb.DMatrix(Xc)
                    leaf_pred_c = booster.predict(dmatrix_c, pred_leaf=True)

                    # 计算叶子IDF权重
                    for tree_idx in range(leaf_pred.shape[1]):
                        leaf_counts = np.bincount(leaf_pred_c[:, tree_idx])
                        total = len(leaf_pred_c)
                        for i in range(len(Ai)):
                            leaf_id = leaf_pred[i, tree_idx]
                            freq = leaf_counts[leaf_id] / total if leaf_id < len(leaf_counts) else 0
                            s_leaf[i] += np.log(1.0 / (freq + 1e-6))
                else:
                    # 使用A自身计算叶子稀有度
                    for tree_idx in range(leaf_pred.shape[1]):
                        leaf_counts = np.bincount(leaf_pred[:, tree_idx])
                        total = len(leaf_pred)
                        for i in range(len(Ai)):
                            leaf_id = leaf_pred[i, tree_idx]
                            freq = leaf_counts[leaf_id] / total
                            s_leaf[i] += np.log(1.0 / (freq + 1e-6))

                if s_leaf.max() > 0:
                    s_leaf = (s_leaf - s_leaf.min()) / (s_leaf.max() - s_leaf.min() + eps)
            except:
                s_leaf = np.zeros(len(Ai))

            # 动态权重分配：基于特征重要性
            weights = np.array([0.35, 0.30, 0.25, 0.10])  # loss, conf, correct, leaf
            scores = (weights[0] * s_loss + weights[1] * s_conf +
                      weights[2] * s_correct + weights[3] * s_leaf)

            return scores

        except Exception as e:
            print(f"DI attack failed: {e}")
            return np.zeros(len(pd.read_csv(A_path)), dtype=np.float32)


class AdaptiveHybridAttack:
    """自适应混合攻击：动态调整CI和DI的权重"""

    def __init__(self, base_ci_weight: float = 0.6, adaptability_threshold: float = 0.3):
        self.ci_attack = EnhancedCIAttack()
        self.di_attack = EnhancedDIAttack()
        self.base_ci_weight = base_ci_weight
        self.adaptability_threshold = adaptability_threshold

    def compute_confidence_scores(self, ci_scores: np.ndarray, di_scores: np.ndarray) -> Tuple[float, float]:
        """计算CI和DI的置信度以动态调整权重"""
        # CI置信度：基于分数分布的区分度
        ci_confidence = np.std(ci_scores) / (np.mean(ci_scores) + 1e-8) if np.any(ci_scores) else 0

        # DI置信度：基于分数范围和分布
        di_confidence = (np.percentile(di_scores, 75) - np.percentile(di_scores, 25)) if np.any(di_scores) else 0

        total_confidence = ci_confidence + di_confidence + 1e-8
        dynamic_ci_weight = ci_confidence / total_confidence
        dynamic_di_weight = di_confidence / total_confidence

        # 平滑调整，避免剧烈变化
        final_ci_weight = 0.7 * self.base_ci_weight + 0.3 * dynamic_ci_weight
        final_di_weight = 1.0 - final_ci_weight

        return final_ci_weight, final_di_weight

    def attack(self, team: str) -> np.ndarray:
        """执行混合攻击"""
        A_path = DATA / f"AA{team}.csv"
        C_path = DATA / f"CC{team}.csv"
        D_path = DATA / f"DD{team}.json"

        have_ci = A_path.exists() and C_path.exists()
        have_di = A_path.exists() and D_path.exists()

        if not have_ci and not have_di:
            print(f"[Enhanced] Team {team}: No data available, returning zeros")
            return np.zeros(NUM_ROWS, dtype=int)

        # 计算CI分数（如果可用）
        ci_scores = np.zeros(NUM_ROWS, dtype=np.float32)
        if have_ci:
            try:
                print(f"[Enhanced] Team {team}: Computing CI scores...")
                ci_scores = self.ci_attack.stability_voting(A_path, C_path)
            except Exception as e:
                print(f"[Enhanced] CI attack failed for team {team}: {e}")

        # 计算DI分数（如果可用）
        di_scores = np.zeros(NUM_ROWS, dtype=np.float32)
        if have_di:
            try:
                print(f"[Enhanced] Team {team}: Computing DI scores...")
                C_path_for_di = C_path if have_ci else None
                di_scores = self.di_attack.multi_feature_scoring(D_path, A_path, C_path_for_di)
            except Exception as e:
                print(f"[Enhanced] DI attack failed for team {team}: {e}")

        # 动态权重调整
        if have_ci and have_di:
            ci_weight, di_weight = self.compute_confidence_scores(ci_scores, di_scores)
            print(f"[Enhanced] Team {team}: Dynamic weights - CI: {ci_weight:.3f}, DI: {di_weight:.3f}")
        elif have_ci:
            ci_weight, di_weight = 1.0, 0.0
        else:
            ci_weight, di_weight = 0.0, 1.0

        # 分数融合
        combined_scores = ci_weight * ci_scores + di_weight * di_scores

        # 选择top-k
        if np.any(combined_scores):
            top_indices = np.argsort(combined_scores)[-TARGET_ONES:]
            result = np.zeros(NUM_ROWS, dtype=int)
            result[top_indices] = 1
            return result
        else:
            # Fallback: 随机选择
            print(f"[Enhanced] Team {team}: Using random fallback")
            result = np.zeros(NUM_ROWS, dtype=int)
            result[self.ci_attack.rng.choice(NUM_ROWS, size=TARGET_ONES, replace=False)] = 1
            return result


def load_col_csv(path: Path) -> np.ndarray:
    """加载列CSV文件"""
    s = pd.read_csv(path, header=None).iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    s = (s > 0.5).astype(int).values
    if len(s) != NUM_ROWS:
        raise ValueError(f"{path} rows={len(s)} != {NUM_ROWS}")
    return s


def main():
    ap = argparse.ArgumentParser(description="Enhanced Hybrid MIA Attack")
    ap.add_argument("my", help="Your team number, e.g., 07")
    ap.add_argument("--own_zero", action="store_true", help="Set your own column to zeros")
    ap.add_argument("--overwrite", "-o", action="store_true", help="Overwrite existing files")
    ap.add_argument("--force_zero", type=str, default="", help="Comma-separated team numbers to force zero")
    ap.add_argument("--base_ci_weight", type=float, default=0.6, help="Base weight for CI attack")

    args = ap.parse_args()

    my_team = args.my.zfill(2)
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"F{my_team}.csv"

    if out_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{out_csv} already exists. Use -o to overwrite.")

    # 初始化攻击器
    attacker = AdaptiveHybridAttack(base_ci_weight=args.base_ci_weight)

    # 构建结果DataFrame
    teams = [f"{i:02d}" for i in range(1, 25)]
    result_df = pd.DataFrame(index=range(NUM_ROWS), columns=teams)

    force_zero_teams = set([t.zfill(2) for t in args.force_zero.split(",") if t.strip()])

    for team in teams:
        print(f"[Enhanced] Processing team {team}...")

        if team in force_zero_teams:
            result_df[team] = 0
            print(f"[Enhanced] Team {team} forced to zero")
            continue

        if team == my_team:
            result_df[team] = 0 if args.own_zero else ""
            continue

        try:
            result_col = attacker.attack(team)
            result_df[team] = result_col
            print(f"[Enhanced] Team {team} completed - ones: {result_col.sum()}")
        except Exception as e:
            print(f"[Enhanced] Team {team} failed: {e}, using random fallback")
            rng = np.random.default_rng(2025 + int(team))
            fallback = np.zeros(NUM_ROWS, dtype=int)
            fallback[rng.choice(NUM_ROWS, size=TARGET_ONES, replace=False)] = 1
            result_df[team] = fallback

    # 保存结果
    result_df.to_csv(out_csv, index=False, header=False, na_rep="")

    # 创建ZIP文件
    from zipfile import ZipFile, ZIP_DEFLATED
    id_file = OUT / "id.txt"
    with open(id_file, "w") as f:
        f.write(my_team)

    zip_path = OUT / f"F{my_team}.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        zf.write(out_csv, out_csv.name)
        zf.write(id_file, "id.txt")

    print(f"[Enhanced] Successfully saved {out_csv} and {zip_path}")


if __name__ == "__main__":
    main()