# attack/attack_enhanced.py
# ============== 使用说明（新增 CLI）==============
# --jobs N             并行队列（默认1；>=4 推荐）
# --ci-rounds R        CI 稳定投票轮数（默认12，原30）
# --ci-subspace RATIO  子空间特征比例（默认0.6）
# --ci-knn K           每轮k近邻（默认5）
# --ci-csample R       C侧子样本比例(0~1，默认0.3；设1.0为全量)
# --ci-metric m        "l1"或"cosine"（cosine 通过L2归一化+欧氏实现）
# =================================================
import argparse, sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

CUR = Path(__file__).resolve().parent
ROOT = CUR.parent
DATA = ROOT / "attack_data"
OUT = ROOT / "outputs_enhanced"

sys.path.insert(0, str(CUR))
try:
    from mia import build_feature_matrices
except ImportError:
    build_feature_matrices = None

try:
    sys.path.insert(0, str(ROOT / "analysis"))
    from xgbt_train import build_X
except ImportError:
    build_X = None

NUM_ROWS = 100_000
TARGET_ONES = 10_000
TARGET = "stroke_flag"


# ---------------------- 工具函数 ----------------------
def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    # 列名统一：去空白、小写
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _numeric_ratio(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return (~x.isna()).mean()


# ---------------------- 特征编码器 ----------------------
class FeatureEncoder:
    """
    一次性对 A/C 对齐编码：
      - 数值列：联合(A∪C)的中位数/IQR鲁棒标准化
      - 类别列：联合字典的有序编码（保持一致），低基数<=30做 one-hot；
                高基数做“Top-N保留 + 其他桶”
    """

    def __init__(self, low_card_max=30, topn_high_card=20, random_state=2025):
        self.low_card_max = low_card_max
        self.topn_high_card = topn_high_card
        self.rng = np.random.default_rng(random_state)
        self.col_info = []  # [(col, kind, params), ...] kind in {"num","onehot","bucket"}

    def fit(self, A: pd.DataFrame, C: pd.DataFrame, exclude: Optional[List[str]] = None):
        A = _clean_cols(A)
        C = _clean_cols(C)
        if exclude is None: exclude = []
        common = [c for c in A.columns if c in C.columns and c not in set(exclude)]
        self.col_info.clear()
        for col in common:
            sA = A[col].astype(str).str.strip()
            sC = C[col].astype(str).str.strip()
            rA = _numeric_ratio(sA)
            rC = _numeric_ratio(sC)
            # 判定为数值列的条件：两侧都有足够比例可数值化
            if rA > 0.8 and rC > 0.8:
                z = pd.to_numeric(pd.concat([sA, sC], ignore_index=True), errors="coerce")
                med = np.nanmedian(z)
                q1, q3 = np.nanpercentile(z, [25, 75])
                iqr = (q3 - q1) if (q3 > q1) else (np.nanstd(z) + 1e-6)
                self.col_info.append((col, "num", {"med": float(med), "scale": float(iqr if iqr > 1e-6 else 1.0)}))
            else:
                vals = pd.concat([sA, sC], ignore_index=True)
                vc = vals.value_counts(dropna=False)
                uniq = len(vc)
                if uniq <= self.low_card_max:
                    # low-card one-hot
                    cats = list(vc.index)
                    self.col_info.append((col, "onehot", {"cats": cats}))
                else:
                    # high-card: Top-N + "__other__"
                    cats = list(vc.index[:self.topn_high_card])
                    self.col_info.append((col, "bucket", {"cats": cats, "other": "__other__"}))
        return self

    def transform(self, D: pd.DataFrame) -> np.ndarray:
        D = _clean_cols(D)
        feats = []
        for col, kind, prm in self.col_info:
            if col not in D.columns:
                # 缺列：填零
                if kind == "num":
                    feats.append(np.zeros((len(D), 1), dtype=np.float32))
                elif kind == "onehot":
                    feats.append(np.zeros((len(D), len(prm["cats"])), dtype=np.float32))
                else:
                    feats.append(np.zeros((len(D), len(prm["cats"]) + 1), dtype=np.float32))
                continue
            s = D[col].astype(str).str.strip()
            if kind == "num":
                x = pd.to_numeric(s, errors="coerce").fillna(prm["med"]).to_numpy(dtype=np.float32)
                x = (x - prm["med"]) / (prm["scale"] if prm["scale"] > 1e-6 else 1.0)
                feats.append(x.reshape(-1, 1))
            elif kind == "onehot":
                cats = prm["cats"]
                idx = {v: i for i, v in enumerate(cats)}
                arr = np.zeros((len(D), len(cats)), dtype=np.float32)
                for i, v in enumerate(s):
                    j = idx.get(v, None)
                    if j is not None: arr[i, j] = 1.0
                feats.append(arr)
            else:
                cats = prm["cats"]
                idx = {v: i for i, v in enumerate(cats)}
                arr = np.zeros((len(D), len(cats) + 1), dtype=np.float32)
                other_idx = len(cats)
                for i, v in enumerate(s):
                    j = idx.get(v, other_idx)
                    arr[i, j] = 1.0
                feats.append(arr)
        if not feats:
            return np.zeros((len(D), 0), dtype=np.float32)
        return np.concatenate(feats, axis=1).astype(np.float32)

    def fit_transform_pair(self, A: pd.DataFrame, C: pd.DataFrame, exclude: Optional[List[str]] = None) -> Tuple[
        np.ndarray, np.ndarray]:
        self.fit(A, C, exclude=exclude)
        return self.transform(A), self.transform(C)


# ---------------------- CI 攻击 ----------------------
class EnhancedCIAttack:
    """更快更稳的 CI：一次编码，多轮子空间；支持 C 端子采样；修复子空间对齐 bug"""

    def __init__(self, random_state: int = 2025):
        self.rng = np.random.default_rng(random_state)
        self.encoder = FeatureEncoder(random_state=random_state)

    def _prepare_features(self, A_df: pd.DataFrame, C_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # 优先用外部提供的特征构建器（如果你们 mia.build_feature_matrices 很强）
        if build_feature_matrices is not None:
            try:
                X1, X2 = build_feature_matrices(A_df, C_df)
                return X1.values.astype(np.float32), X2.values.astype(np.float32)
            except:
                pass
        # 退化：我们自己的稳健编码
        return self.encoder.fit_transform_pair(A_df, C_df, exclude=[TARGET])

    @staticmethod
    def _l2_normalize(X: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / n

    def stability_voting(
            self, A_path: Path, C_path: Path,
            n_rounds: int = 12, subspace_ratio: float = 0.6,
            k_neighbors: int = 5, c_sample_ratio: float = 0.3,
            metric: str = "cosine"
    ) -> np.ndarray:
        from sklearn.neighbors import NearestNeighbors

        A_df = pd.read_csv(A_path, dtype=str, keep_default_na=False)
        C_df = pd.read_csv(C_path, dtype=str, keep_default_na=False)

        X1, X2 = self._prepare_features(A_df, C_df)
        m = len(A_df)
        if X1.shape[1] == 0 or m == 0:
            return np.zeros(m, dtype=np.float32)

        # 余弦距离 → L2 归一化 + 欧氏距离（快）
        if metric == "cosine":
            X1n = self._l2_normalize(X1)
            X2n = self._l2_normalize(X2)
            metric_knn = "euclidean"
        else:
            X1n, X2n = X1, X2
            metric_knn = "manhattan"  # L1

        # C 侧子采样（极大提速）
        if 0 < c_sample_ratio < 1.0:
            c_idx = self.rng.choice(len(X2n), size=int(len(X2n) * c_sample_ratio), replace=False)
            X2n_eff = X2n[c_idx]
            c_weight = float(len(X2n)) / float(len(X2n_eff))  # 估计补偿
        else:
            X2n_eff = X2n
            c_weight = 1.0

        scores = np.zeros(m, dtype=np.float32)
        n_features = X1n.shape[1]
        subspace = max(3, int(n_features * subspace_ratio))

        # 统一构建 NN 对象（每轮换子空间需要重建）
        for _ in range(n_rounds):
            if n_features > subspace:
                feat_idx = self.rng.choice(n_features, size=subspace, replace=False)
                A_sub = X1n[:, feat_idx]
                C_sub = X2n_eff[:, feat_idx]
            else:
                A_sub, C_sub = X1n, X2n_eff

            # kNN 查询（并行）
            nn = NearestNeighbors(n_neighbors=k_neighbors, metric=metric_knn, n_jobs=-1, algorithm='auto')
            nn.fit(A_sub)
            dist, idx = nn.kneighbors(C_sub, return_distance=True)  # shape: (|C_sub|, k)

            # 权重 1/(d+eps)，向 A 的 index 聚合
            w = 1.0 / (dist + 1e-8)
            # 累加（使用 bincount 一次性聚合）
            add_idx = idx.ravel()
            add_w = (w * c_weight).ravel()
            bin_w = np.bincount(add_idx, weights=add_w, minlength=m)
            scores += bin_w.astype(np.float32)

        # 归一化到[0,1]
        if scores.max() > 0:
            smin, smax = scores.min(), scores.max()
            scores = (scores - smin) / (smax - smin + 1e-12)
        return scores


# ---------------------- DI 攻击 ----------------------
class EnhancedDIAttack:
    """多特征融合 + 叶子稀有度向量化 + 置信度校准"""

    def __init__(self):
        pass

    def _align_features(self, X: pd.DataFrame, booster) -> pd.DataFrame:
        if hasattr(booster, 'feature_names') and booster.feature_names:
            model_features = list(booster.feature_names)  # 保序
            for feat in model_features:
                if feat not in X.columns:
                    X[feat] = 0
            # 只保留模型需要的列并排序
            X = X[model_features]
        return X

    def multi_feature_scoring(self, D_path: Path, A_path: Path, C_path: Optional[Path] = None) -> np.ndarray:
        try:
            import xgboost as xgb
        except Exception as e:
            print(f"xgboost not available: {e}")
            return np.zeros(NUM_ROWS, dtype=np.float32)

        Ai = pd.read_csv(A_path, dtype=str, keep_default_na=False)
        y = pd.to_numeric(Ai.get(TARGET, 0), errors='coerce').fillna(0).astype(np.int32).to_numpy()

        # 构特征
        if build_X is not None:
            try:
                X = build_X(Ai, TARGET)
            except:
                fea_cols = [c for c in Ai.columns if c != TARGET]
                X = Ai[fea_cols].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0))
        else:
            fea_cols = [c for c in Ai.columns if c != TARGET]
            X = Ai[fea_cols].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0))

        # 加载 XGB 模型
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(D_path))
        X = self._align_features(X, booster)
        dA = xgb.DMatrix(X, feature_names=list(X.columns))

        # 预测概率
        pred = booster.predict(dA)
        eps = 1e-12
        p = np.clip(pred, eps, 1 - eps)

        # 1) 负对数似然
        nll = - (y * np.log(p) + (1 - y) * np.log(1 - p))
        s_loss = 1.0 - (nll - nll.min()) / (nll.max() - nll.min() + eps)

        # 2) 置信度
        conf = np.where(y == 1, p, 1 - p)
        s_conf = (conf - conf.min()) / (conf.max() - conf.min() + eps)

        # 3) 预测正确性
        s_correct = ((p >= 0.5).astype(np.int32) == y).astype(np.float32)

        # 4) 叶子稀有度（向量化）
        try:
            leafA = booster.predict(dA, pred_leaf=True)  # shape: (N, ntrees)
            if C_path and Path(C_path).exists():
                Ci = pd.read_csv(C_path, dtype=str, keep_default_na=False)
                if build_X is not None:
                    try:
                        Xc = build_X(Ci, TARGET)
                    except:
                        fea_cols_c = [c for c in Ci.columns if c != TARGET]
                        Xc = Ci[fea_cols_c].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0))
                else:
                    fea_cols_c = [c for c in Ci.columns if c != TARGET]
                    Xc = Ci[fea_cols_c].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0))
                Xc = self._align_features(Xc, booster)
                dC = xgb.DMatrix(Xc, feature_names=list(Xc.columns))
                leafC = booster.predict(dC, pred_leaf=True)
                # 计算每棵树上 C 的叶子频率，然后映射到 A 的叶子
                s_leaf = np.zeros(leafA.shape[0], dtype=np.float64)
                ntrees = leafA.shape[1]
                for t in range(ntrees):
                    lc = leafC[:, t]
                    max_id = lc.max() if lc.size else 0
                    freq = np.bincount(lc, minlength=int(max_id) + 1)
                    freq = freq / max(1, len(lc))
                    la = leafA[:, t]
                    la = la.clip(0, len(freq) - 1)
                    s_leaf += np.log(1.0 / (freq[la] + 1e-6))
            else:
                s_leaf = np.zeros(leafA.shape[0], dtype=np.float64)
                ntrees = leafA.shape[1]
                for t in range(ntrees):
                    la = leafA[:, t]
                    max_id = la.max() if la.size else 0
                    freq = np.bincount(la, minlength=int(max_id) + 1)
                    freq = freq / max(1, len(la))
                    s_leaf += np.log(1.0 / (freq[la] + 1e-6))
            if np.max(s_leaf) > 0:
                s_leaf = (s_leaf - np.min(s_leaf)) / (np.max(s_leaf) - np.min(s_leaf) + eps)
            s_leaf = s_leaf.astype(np.float32)
        except Exception as e:
            # 叶子不可用时退化
            s_leaf = np.zeros(len(Ai), dtype=np.float32)

        # 融合（权重可按需要调整/网格搜索）
        weights = np.array([0.35, 0.30, 0.25, 0.10], dtype=np.float32)
        scores = (weights[0] * s_loss + weights[1] * s_conf + weights[2] * s_correct + weights[3] * s_leaf).astype(
            np.float32)
        return scores


# ---------------------- 自适应融合 ----------------------
class AdaptiveHybridAttack:
    def __init__(self, base_ci_weight: float = 0.6, adaptability_threshold: float = 0.3,
                 ci_rounds: int = 12, ci_subspace: float = 0.6, ci_knn: int = 5, ci_csample: float = 0.3,
                 ci_metric: str = "cosine"):
        self.ci_attack = EnhancedCIAttack()
        self.di_attack = EnhancedDIAttack()
        self.base_ci_weight = base_ci_weight
        self.adaptability_threshold = adaptability_threshold
        self.ci_rounds = ci_rounds
        self.ci_subspace = ci_subspace
        self.ci_knn = ci_knn
        self.ci_csample = ci_csample
        self.ci_metric = ci_metric

    @staticmethod
    def _safe_std_mean_ratio(x: np.ndarray) -> float:
        m = float(np.mean(x)) + 1e-8
        return float(np.std(x) / m)

    @staticmethod
    def _iqr(x: np.ndarray) -> float:
        return float(np.percentile(x, 75) - np.percentile(x, 25))

    def compute_confidence_scores(self, ci_scores: np.ndarray, di_scores: np.ndarray) -> Tuple[float, float]:
        ci_conf = self._safe_std_mean_ratio(ci_scores) if np.any(ci_scores) else 0.0
        di_conf = self._iqr(di_scores) if np.any(di_scores) else 0.0
        total = ci_conf + di_conf + 1e-8
        dyn_ci = ci_conf / total
        final_ci = 0.7 * self.base_ci_weight + 0.3 * dyn_ci
        final_ci = min(max(final_ci, 0.0), 1.0)
        final_di = 1.0 - final_ci
        return final_ci, final_di

    def attack(self, team: str) -> np.ndarray:
        A_path = DATA / f"AA{team}.csv"
        C_path = DATA / f"CC{team}.csv"
        D_path = DATA / f"DD{team}.json"

        have_ci = A_path.exists() and C_path.exists()
        have_di = A_path.exists() and D_path.exists()

        if not have_ci and not have_di:
            print(f"[Enhanced] Team {team}: No data available, returning zeros")
            return np.zeros(NUM_ROWS, dtype=np.int32)

        ci_scores = np.zeros(NUM_ROWS, dtype=np.float32)
        if have_ci:
            try:
                ci_scores = self.ci_attack.stability_voting(
                    A_path, C_path,
                    n_rounds=self.ci_rounds,
                    subspace_ratio=self.ci_subspace,
                    k_neighbors=self.ci_knn,
                    c_sample_ratio=self.ci_csample,
                    metric=self.ci_metric
                )
            except Exception as e:
                print(f"[Enhanced] CI failed @team {team}: {e}")

        di_scores = np.zeros(NUM_ROWS, dtype=np.float32)
        if have_di:
            try:
                C_for_di = C_path if have_ci else None
                di_scores = self.di_attack.multi_feature_scoring(D_path, A_path, C_for_di)
            except Exception as e:
                print(f"[Enhanced] DI failed @team {team}: {e}")

        if have_ci and have_di:
            w_ci, w_di = self.compute_confidence_scores(ci_scores, di_scores)
            print(f"[Enhanced] Team {team}: Dynamic weights - CI: {w_ci:.3f}, DI: {w_di:.3f}")
        elif have_ci:
            w_ci, w_di = 1.0, 0.0
        else:
            w_ci, w_di = 0.0, 1.0

        combined = w_ci * ci_scores + w_di * di_scores
        if np.any(combined):
            # 只做一次 O(n) 的 argpartition
            topk_idx = np.argpartition(combined, -TARGET_ONES)[-TARGET_ONES:]
            out = np.zeros(NUM_ROWS, dtype=np.int32)
            out[topk_idx] = 1
            return out
        else:
            rng = np.random.default_rng(2025 + int(team))
            out = np.zeros(NUM_ROWS, dtype=np.int32)
            out[rng.choice(NUM_ROWS, size=TARGET_ONES, replace=False)] = 1
            return out


# ---------------------- I/O ----------------------
def load_col_csv(path: Path) -> np.ndarray:
    s = pd.read_csv(path, header=None).iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    s = (s > 0.5).astype(np.int32).values
    if len(s) != NUM_ROWS:
        raise ValueError(f"{path} rows={len(s)} != {NUM_ROWS}")
    return s


# ---------------------- 主程序 ----------------------
def main():
    ap = argparse.ArgumentParser(description="Enhanced Hybrid MIA Attack (Fast)")
    ap.add_argument("my", help="Your team number, e.g., 07")
    ap.add_argument("--own_zero", action="store_true", help="Set your own column to zeros")
    ap.add_argument("--overwrite", "-o", action="store_true", help="Overwrite existing files")
    ap.add_argument("--force_zero", type=str, default="", help="Comma-separated team numbers to force zero")
    ap.add_argument("--base_ci_weight", type=float, default=0.6, help="Base weight for CI attack")

    # 新增加速/控制参数
    ap.add_argument("--jobs", type=int, default=1, help="Parallel jobs across teams")
    ap.add_argument("--ci-rounds", type=int, default=12)
    ap.add_argument("--ci-subspace", type=float, default=0.6)
    ap.add_argument("--ci-knn", type=int, default=5)
    ap.add_argument("--ci-csample", type=float, default=0.3)
    ap.add_argument("--ci-metric", type=str, default="cosine", choices=["cosine", "l1"])

    args = ap.parse_args()
    my_team = args.my.zfill(2)
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"F{my_team}.csv"

    if out_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{out_csv} already exists. Use -o to overwrite.")

    attacker = AdaptiveHybridAttack(
        base_ci_weight=args.base_ci_weight,
        ci_rounds=args.ci_rounds,
        ci_subspace=args.ci_subspace,
        ci_knn=args.ci_knn,
        ci_csample=args.ci_csample,
        ci_metric=("cosine" if args.ci_metric == "cosine" else "manhattan")
    )

    teams = [f"{i:02d}" for i in range(1, 25)]
    force_zero_teams = set([t.zfill(2) for t in args.force_zero.split(",") if t.strip()])

    # 结果矩阵
    result_df = pd.DataFrame(index=range(NUM_ROWS), columns=teams)

    # 任务函数（便于并行）
    def _run_team(team: str):
        if team in force_zero_teams:
            return team, np.zeros(NUM_ROWS, dtype=np.int32), "[Enhanced] forced zero"
        if team == my_team:
            col = np.zeros(NUM_ROWS, dtype=np.int32) if args.own_zero else np.array([""] * NUM_ROWS, dtype=object)
            return team, col, "[Enhanced] my team"
        try:
            col = attacker.attack(team)
            return team, col, f"[Enhanced] ones={int(col.sum())}"
        except Exception as e:
            rng = np.random.default_rng(2025 + int(team))
            fallback = np.zeros(NUM_ROWS, dtype=np.int32)
            fallback[rng.choice(NUM_ROWS, size=TARGET_ONES, replace=False)] = 1
            return team, fallback, f"[Enhanced] fallback due to error: {e}"

    if args.jobs <= 1:
        for team in teams:
            t, col, msg = _run_team(team)
            result_df[t] = col
            print(f"Team {t}: {msg}")
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(_run_team, team): team for team in teams}
            for fut in as_completed(futures):
                t, col, msg = fut.result()
                result_df[t] = col
                print(f"Team {t}: {msg}")

    # 保存
    result_df.to_csv(out_csv, index=False, header=False, na_rep="")

    from zipfile import ZipFile, ZIP_DEFLATED
    id_file = OUT / "id.txt"
    with open(id_file, "w") as f:
        f.write(my_team)
    zip_path = OUT / f"F{my_team}.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        zf.write(out_csv, out_csv.name)
        zf.write(id_file, "id.txt")

    print(f"[Enhanced] Saved {out_csv} and {zip_path}")


if __name__ == "__main__":
    main()
