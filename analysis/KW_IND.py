#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#有用性：按‘年龄组’观察的医疗利用分布差解析
#医疗数据分析脚本：通过Kruskal-Wallis检验分析不同年龄组在医疗利用指标上的分布差异

import argparse
import numpy as np
import pandas as pd
from scipy.stats import kruskal, chi2, norm

TARGET_DEFAULT = "AGE" # 默认年龄列名
DEFAULT_BINS = [0, 18, 45, 65, 75, 200] # 默认年龄分组边界
#0，10，20，30，40，50，60，70，80，90，100
METRICS_DEFAULT = [   # 默认分析的医疗指标
    "encounter_count",
    "num_medications",
    "num_procedures",
    "num_immunizations",
    "num_devices",
]

DEFAULT_P_NORM = "arctan"   # 'arctan' / 'exp' / 'log1p'  # p值归一化方法
DEFAULT_P_SCALE = 10.0   # p值归一化尺度参数
DEFAULT_P_CAP = 300.0    # p值归一化上限

NUM_COLS = ["H_norm", "minus_log10_p_norm", "epsilon2", "eta2_kw", "rank_eta2", "A_pair_avg", "A_pair_sym"]

def find_col_case_insensitive(df: pd.DataFrame, name: str) -> str | None:
    """不区分大小写查找列名"""
    low = name.lower()
    for c in df.columns:
        if c.lower() == low:
            return c
    return None

def make_age_groups_by_custom_bins(age_series: pd.Series, custom_bins: list[float]) -> pd.Series:
    """根据自定义边界创建年龄分组"""
    age = pd.to_numeric(age_series, errors="coerce") # 转换为数值，无效值转为NaN
    bins = np.array(sorted(custom_bins), dtype=float)
    if len(bins) < 3:
        raise ValueError("--custom-bins は 3 個以上の境界が必要です（>=2 群）。")
    # 创建分组标签
    labels = []
    for i in range(len(bins) - 1):
        a, b = bins[i], bins[i + 1]
        labels.append(f"{int(a)}+" if i == len(bins) - 2 else f"{int(a)}–{int(b)-1}")
    # 调整边界以确保包含所有数据
    finite_max = np.nanmax(age.values) if np.isfinite(np.nanmax(age.values)) else bins[-1]
    extended = bins.copy()
    extended[-1] = max(bins[-1], finite_max) + 1e-9
    return pd.cut(age, bins=extended, right=False, labels=labels, include_lowest=True)

def chi2_logp_safe(H: float, dfree: int) -> tuple[float, str]:
    """安全计算卡方检验的p值（避免数值溢出）"""
    # 首先尝试精确计算
    logp = chi2.logsf(H, dfree)
    if np.isfinite(logp):
        mlog10p = -logp / np.log(10.0)
        p_str = f"1e-{mlog10p:.1f}" if mlog10p > 300 else f"{np.exp(logp):.3e}"
        return float(mlog10p), p_str
    # 使用正态近似
    v = float(dfree)
    w = (H / v) ** (1.0 / 3.0)
    mu = 1.0 - 2.0 / (9.0 * v)
    sigma = np.sqrt(2.0 / (9.0 * v))
    Z = (w - mu) / sigma
    logp_norm = norm.logsf(abs(Z)) + np.log(2.0)
    if np.isfinite(logp_norm):
        mlog10p = -logp_norm / np.log(10.0)
        p_str = f"1e-{mlog10p:.1f}" if mlog10p > 300 else f"{np.exp(logp_norm):.3e}"
        return float(mlog10p), p_str

    # 极端情况下的近似
    Z = abs(Z)
    mlog10p = (Z * Z) / (2.0 * np.log(10.0)) + (np.log(Z) + 0.5 * np.log(2.0 * np.pi)) / np.log(10.0)
    p_str = f"1e-{mlog10p:.1f}"
    return float(mlog10p), p_str

def eta2_kw(H: float, n_eff: int, k_used: int) -> float:
    if n_eff <= 1:
        return 0.0
    val = (H - (k_used - 1.0)) / (n_eff - 1.0)
    return float(np.clip(val, 0.0, 1.0))

def rank_eta2(y: np.ndarray, g_codes: np.ndarray, G: int) -> float:
    """基于排名的eta平方计算"""
    ranks = pd.Series(y).rank(method="average").to_numpy() # 计算排名
    ybar = ranks.mean()
    ssb = 0.0
    # 计算组间平方和
    for c in range(G):
        mask = (g_codes == c)
        if mask.any():
            n_c = mask.sum()
            m_c = ranks[mask].mean()
            ssb += n_c * (m_c - ybar) ** 2
    sst = ((ranks - ybar) ** 2).sum()  # 总平方和
    return float(ssb / sst) if sst > 0 else 0.0

def vargha_delaney_A(x: np.ndarray, y: np.ndarray) -> float:
    """计算Vargha-Delaney A效应量（两组比较）"""
    s = np.concatenate([x, y])
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = len(x)
    r1 = r[:n1].sum()
    A = (r1 - n1 * (n1 + 1) / 2.0) / (n1 * len(y))
    return float(np.clip(A, 0.0, 1.0))

def multi_group_A_metrics(groups: list[np.ndarray]) -> tuple[float, float]:
    """多组比较的A效应量指标"""
    A_sum = 0.0
    Asym_sum = 0.0
    w_sum = 0.0
    # 对所有组对进行计算
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            xi, xj = groups[i], groups[j]
            if len(xi) == 0 or len(xj) == 0:
                continue
            A = vargha_delaney_A(xi, xj)
            w = len(xi) * len(xj)  # 权重为样本量乘积
            A_sum += w * A
            Asym_sum += w * (2.0 * abs(A - 0.5)) # 对称性度量
            w_sum += w
    if w_sum == 0:
        return 0.5, 0.0
    return float(A_sum / w_sum), float(Asym_sum / w_sum)

def h_max_no_ties(counts: list[int] | np.ndarray) -> float:
    """计算无结值时的最大H统计量"""
    c = np.asarray(counts, dtype=float)
    n = c.sum()
    if n <= 1 or (c <= 0).any():
        return 0.0
    prefix = np.concatenate(([0.0], np.cumsum(c[:-1])))
    Rbar = prefix + (c + 1.0) / 2.0  # 期望排名
    overall = (n + 1.0) / 2.0
    ssb = np.sum(c * (Rbar - overall) ** 2)  # 组间平方和
    Hmax = (12.0 / (n * (n + 1.0))) * ssb
    return float(max(Hmax, 0.0))

def normalize_mlog10p(x, method=DEFAULT_P_NORM, scale=DEFAULT_P_SCALE, cap=DEFAULT_P_CAP):
    """对-log10(p)进行0-1归一化"""
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x) & (x >= 0), x, 0.0)
    if method == "arctan":
        return (2.0 / np.pi) * np.arctan(x / float(scale)) # 反正切归一化
    elif method == "exp":
        return 1.0 - np.exp(-x / float(scale))# 指数归一化
    else:  # "log1p"
        cap = float(cap)
        return np.log1p(np.minimum(x, cap)) / np.log1p(cap)# 对数归一化

# ==== 对一个CSV计算KW指标 ====

def compute_kw_table(df:pd.DataFrame, # csv_path: str,
                     age_col_name: str,
                     metrics_csv: str,
                     custom_bins_str: str,
                     min_per_group: int,
                     p_norm: str, p_scale: float, p_cap: float) -> pd.DataFrame:
    """计算Kruskal-Wallis检验结果表"""

    # 查找年龄列
    age_col = find_col_case_insensitive(df, age_col_name)
    if age_col is None:
        # raise SystemExit(f"[{csv_path}] 年龄列 {age_col_name} 未找到。")
        raise SystemExit(f"年齢列 {age_col_name} が見つかりません。")

    # 解析指标列
    metric_names = [m.strip() for m in metrics_csv.split(",") if m.strip()]
    metrics, not_found = [], []
    for m in metric_names:
        col = find_col_case_insensitive(df, m)
        (metrics if col is not None else not_found).append(col or m)


    if not metrics:
        # 如果都没有，就为空表（后续可以用0填补差异）
        return pd.DataFrame(columns=["metric"] + NUM_COLS)

    # 创建年龄分组
    custom_bins = DEFAULT_BINS if not custom_bins_str.strip() else [float(x) for x in custom_bins_str.split(",") if x.strip()]
    groups_ser = make_age_groups_by_custom_bins(df[age_col], custom_bins)

    rows = []
    for m in metrics:
        if m not in df.columns:
            # 没有的指标用0填充（用于差分）
            rows.append({"metric": m, **{k: 0.0 for k in NUM_COLS}})
            continue

        # 数据预处理
        y_all = pd.to_numeric(df[m], errors="coerce")
        mask = y_all.notna() & groups_ser.notna()
        y = y_all[mask].to_numpy(dtype=float)
        g = pd.Categorical(groups_ser[mask])
        labels = list(g.categories.astype(str))
        k = len(labels)

        #分组数据
        grp_vals = [y[g.codes == i] for i in range(k)]
        used_vals = [arr for arr in grp_vals if len(arr) >= min_per_group]
        n_eff = sum(len(arr) for arr in used_vals)
        k_used = len(used_vals)

        if k_used < 2:
            rows.append({"metric": m, **{k: 0.0 for k in NUM_COLS}})
            continue

        #执行Kruskal-Wallis检验
        try:
            H, _ = kruskal(*used_vals)
        except Exception:
            rows.append({"metric": m, **{k: 0.0 for k in NUM_COLS}})
            continue

        # 计算统计量
        dfree = k_used - 1
        mlog10p, _ = chi2_logp_safe(float(H), dfree)

        # 计算各种效应量
        eps = (H - dfree) / (n_eff - dfree) if (n_eff - dfree) > 0 else 0.0
        eps = float(np.clip(eps, 0.0, 1.0))
        eta = eta2_kw(float(H), int(n_eff), int(k_used))
        g_codes = np.concatenate([np.full(len(arr), i, dtype=int) for i, arr in enumerate(used_vals)])
        r_eta = rank_eta2(y=np.concatenate(used_vals), g_codes=g_codes, G=k_used)
        A_avg, A_sym = multi_group_A_metrics(used_vals)

        # H统计量归一化（H/Hmax）
        Hmax = h_max_no_ties([len(arr) for arr in used_vals])
        H_scaled_max = float(H / Hmax) if Hmax > 0 else 0.0
        H_scaled_max = float(np.clip(H_scaled_max, 0.0, 1.0))

        # minus_log10_p 的 0-1 归一化（不易饱和）  p值归一化
        pnorm = float(normalize_mlog10p(mlog10p, method=p_norm, scale=p_scale, cap=p_cap))

        rows.append({
            "metric": m,
            "H_norm": H_scaled_max,
            "minus_log10_p_norm": pnorm,
            "epsilon2": eps,
            "eta2_kw": eta,
            "rank_eta2": float(r_eta),
            "A_pair_avg": A_avg,
            "A_pair_sym": A_sym
        })

    # 结果整理
    out = pd.DataFrame(rows, columns=["metric"] + NUM_COLS)
    # 缺失用0填充
    for c in NUM_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out = out.sort_values("metric", kind="mergesort").reset_index(drop=True)
    return out

def main():
    """主函数：处理命令行参数并执行分析"""
    ap = argparse.ArgumentParser(
        description="Kruskal–Wallis（稳定p值・0～1效应量・H归一化）"
    )
    ap.add_argument("csv", help="输入CSV文件")
    ap.add_argument("--age-col", default=TARGET_DEFAULT, help="年齢列名（默认: AGE）")
    ap.add_argument("--metrics", default=",".join(METRICS_DEFAULT),
                    help="分析的指标列（逗号分隔）: " + ",".join(METRICS_DEFAULT))
    ap.add_argument("--custom-bins", default="",
                    help=f"自定义年龄分组边界（例: 0,18,45,65,75,200）。未指定なら {DEFAULT_BINS} を使用")
    ap.add_argument("--min-per-group", type=int, default=2, help="每组最小样本数（既定: 2）")
    ap.add_argument("--p-norm", choices=["arctan", "exp", "log1p"], default=DEFAULT_P_NORM,
                    help="minus_log10_p の 0–1 正規化方式（既定: arctan）")
    ap.add_argument("--p-scale", type=float, default=DEFAULT_P_SCALE,
                    help="arctan/exp のスケール（大きいほどゆっくり1に近づく）既定: 10")
    ap.add_argument("--p-cap", type=float, default=DEFAULT_P_CAP,
                    help="log1p 正規化の上限（既定: 300）")
    args = ap.parse_args()

    # 读取数据
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)

    # 验证列存在性
    age_col = find_col_case_insensitive(df, args.age_col)
    if age_col is None:
        raise SystemExit(f"年龄列 {args.age_col} 未找到。")

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    metrics, not_found = [], []
    for m in metric_names:
        col = find_col_case_insensitive(df, m)
        (metrics if col is not None else not_found).append(col or m)
    if not metrics:
        raise SystemExit("未找到可分析的指标列。--metrics を確認してください。")
    if not_found:
        print(f"※ 未找到的列: {', '.join(not_found)}")

    # 年龄分组
    custom_bins = DEFAULT_BINS if not args.custom_bins.strip() else [float(x) for x in args.custom_bins.split(",") if x.strip()]
    groups_ser = make_age_groups_by_custom_bins(df[age_col], custom_bins)

    rows = []
    #对每个指标进行分析
    for m in metrics:
        # [数据预处理和统计计算...]
        # 这部分与compute_kw_table类似，但包含更详细的输出信息
        y_all = pd.to_numeric(df[m], errors="coerce")
        mask = y_all.notna() & groups_ser.notna()
        y = y_all[mask].to_numpy(dtype=float)
        g = pd.Categorical(groups_ser[mask])
        labels = list(g.categories.astype(str))
        k = len(labels)

        grp_vals = [y[g.codes == i] for i in range(k)]
        sizes = [len(v) for v in grp_vals]
        used_vals = [arr for arr in grp_vals if len(arr) >= args.min_per_group]
        used_labels = [lab for lab, arr in zip(labels, grp_vals) if len(arr) >= args.min_per_group]
        n_eff = sum(len(arr) for arr in used_vals)
        k_used = len(used_vals)

        if k_used < 2:
            rows.append({
                "metric": m,
                "group_sizes": "; ".join(f"{lab}:{sz}" for lab, sz in zip(labels, sizes)),
                "H_norm": np.nan,
                "p_value": "NA",
                "minus_log10_p_norm": np.nan,
                "epsilon2": np.nan, "eta2_kw": np.nan, "rank_eta2": np.nan,
                "A_pair_avg": np.nan, "A_pair_sym": np.nan,
#                "H_cdf": np.nan, "H_scaled_max": np.nan,
                "note": f"有効群不足（min_per_group={args.min_per_group}）"
            })
            continue

        # Kruskal–Wallis
        try:
            H, _ = kruskal(*used_vals)
        except Exception:
            rows.append({
                "metric": m,
                "group_sizes": "; ".join(f"{lab}:{sz}" for lab, sz in zip(labels, sizes)),
                "H_norm": np.nan,
                "p_value": "NA",
                "minus_log10_p_norm": np.nan,
                "epsilon2": np.nan, "eta2_kw": np.nan, "rank_eta2": np.nan,
                "A_pair_avg": np.nan, "A_pair_sym": np.nan,
#                "H_cdf": np.nan, "H_scaled_max": np.nan,
                "note": "kruskal計算エラー"
            })
            continue

        dfree = k_used - 1
        mlog10p, p_str = chi2_logp_safe(float(H), dfree)

        # 効果量
        eps = (H - dfree) / (n_eff - dfree) if (n_eff - dfree) > 0 else 0.0
        eps = float(np.clip(eps, 0.0, 1.0))
        eta = eta2_kw(float(H), int(n_eff), int(k_used))
        g_codes = np.concatenate([np.full(len(arr), i, dtype=int) for i, arr in enumerate(used_vals)])
        r_eta = rank_eta2(y=np.concatenate(used_vals), g_codes=g_codes, G=k_used)
        A_avg, A_sym = multi_group_A_metrics(used_vals)

        # H の 0-1 正規化
        H_cdf = float(chi2.cdf(H, dfree))
        Hmax = h_max_no_ties([len(arr) for arr in used_vals])
        H_scaled_max = float(H / Hmax) if Hmax > 0 else 0.0
        H_scaled_max = float(np.clip(H_scaled_max, 0.0, 1.0))
        H_norm = H_scaled_max

        # minus_log10_p の 0-1 正規化（飽和しにくい）
        pnorm = float(normalize_mlog10p(
            mlog10p, method=args.p_norm, scale=args.p_scale, cap=args.p_cap
        ))

        rows.append({
            "metric": m,
            "group_sizes": "; ".join(f"{lab}:{len(arr)}" for lab, arr in zip(used_labels, used_vals)),
            "H_norm": H_norm,
            "p_value": p_str,
            "minus_log10_p_norm": pnorm,
            "epsilon2": eps,
            "eta2_kw": eta,
            "rank_eta2": float(r_eta),
            "A_pair_avg": A_avg,
            "A_pair_sym": A_sym,
#            "H_cdf": H_cdf,
#            "H_scaled_max": H_scaled_max,
            "note": ""
        })

    # ★ 输出结果：H と minus_log10_p は出力しない
    out = pd.DataFrame(rows, columns=[
        "metric",
        "group_sizes",
#        "H",
        "H_norm",
#        "p_value",
#        "minus_log10_p",
        "minus_log10_p_norm",
        "epsilon2", "eta2_kw", "rank_eta2", "A_pair_avg", "A_pair_sym",
#        "H_cdf", "H_scaled_max",
        "note"
    ])

    pd.set_option("display.max_columns", None)
    with pd.option_context('display.float_format', lambda x: f"{x:.6g}"):
        print("\n=== Kruskal–Wallis + 0–1 Effect Measures + H Normalization (custom age bins) ===")
        print(out.to_string(index=False))

if __name__ == "__main__":
    main()
