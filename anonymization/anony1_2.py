#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anony2_strong.py
- Class-conditional Gaussian Copula synthesis (within groups) + integer-safe quantile projection
- Optional LR logit alignment along continuous-feature subspace (orthogonal perturbation avoided)
- Preserves KW age-bin structure by grouping with AGEBIN; preserves p(X|Y) by conditioning on asthma_flag (+ demographics if k>=kmin)
- Strong break of record linkability (no row-wise reuse; all numeric values re-sampled), while keeping marginals/correlations close.

Usage:
  python anony2_strong.py --bi data/BB07_1.csv --out CC07_1_XX.csv \
      --group AGEBIN,asthma_flag,GENDER,RACE,ETHNICITY --kmin 40 --seed 7 --logit-align
"""
import argparse, numpy as np, pandas as pd
from scipy.stats import norm
import statsmodels.api as sm

NUM_COLS = [
    "AGE","encounter_count","num_procedures","num_medications",
    "num_immunizations","num_allergies","num_devices",
    "mean_systolic_bp","mean_diastolic_bp","mean_bmi","mean_weight"
]
INT_COLS = ["AGE","encounter_count","num_procedures","num_medications","num_immunizations","num_allergies","num_devices"]
CAT_COLS = ["GENDER","RACE","ETHNICITY"]
FLAG_COLS = ["asthma_flag","stroke_flag","obesity_flag","depression_flag"]

DEFAULT_BINS = [0,18,45,65,75,200]
DEFAULT_LABELS = ["0-17","18-44","45-64","65-74","75+"]

def agebin(s):
    return pd.cut(pd.to_numeric(s, errors="coerce"),
                  bins=DEFAULT_BINS, right=False,
                  labels=DEFAULT_LABELS, include_lowest=True).astype(str)

def nearest_spd(A, eps=1e-6):
    A = (A + A.T) / 2.0
    vals, vecs = np.linalg.eigh(A)
    vals[vals < eps] = eps
    return (vecs * vals) @ vecs.T

def gaussian_copula_group(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, d = values.shape
    X = values.copy().astype(float)
    for j in range(d):
        col = X[:, j]
        m = np.nanmedian(col)
        col[np.isnan(col)] = m
        X[:, j] = col
    U = np.zeros_like(X)
    for j in range(d):
        r = pd.Series(X[:, j]).rank(method="average").to_numpy()
        U[:, j] = (r - 0.5) / (n + 1.0)
    Z = norm.ppf(np.clip(U, 1e-6, 1-1e-6))
    Sigma = np.corrcoef(Z, rowvar=False)
    Sigma = nearest_spd(Sigma)
    Zp = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
    Up = norm.cdf(Zp)
    return Up

def quantile_project_uniforms(Up: np.ndarray, ref_vals: np.ndarray, int_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, d = Up.shape
    Xsyn = np.zeros_like(Up)
    for j in range(d):
        col = pd.to_numeric(ref_vals[:, j], errors="coerce")
        xs = np.sort(col[~np.isnan(col)])
        if len(xs) == 0:
            Xsyn[:, j] = np.nan
            continue
        pos = Up[:, j] * (len(xs)-1)
        if int_mask[j]:
            idx = np.rint(pos).astype(int)
            idx = np.clip(idx, 0, len(xs)-1)
            Xsyn[:, j] = xs[idx]
        else:
            lo = np.floor(pos).astype(int); hi = np.ceil(pos).astype(int)
            frac = pos - lo
            Xsyn[:, j] = xs[lo]*(1-frac) + xs[hi]*frac
    return Xsyn

def build_lr_X(df: pd.DataFrame, target: str):
    y = pd.to_numeric(df[target], errors="coerce")
    X_raw = df.drop(columns=[target], errors="ignore")
    num_cols = [c for c in X_raw.columns if c in NUM_COLS]
    X_num = X_raw[num_cols].apply(pd.to_numeric, errors="coerce") if num_cols else pd.DataFrame(index=df.index)
    cat_cols = [c for c in X_raw.columns if c not in num_cols]
    if cat_cols:
        X_cat = pd.get_dummies(X_raw[cat_cols].replace({"": np.nan}).fillna("(NA)").astype("category"), drop_first=True)
    else:
        X_cat = pd.DataFrame(index=df.index)
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    base_terms = sorted(X.columns.tolist())
    X = X.reindex(columns=base_terms)
    mask = y.notna() & y.isin([0.0, 1.0])
    return X.loc[mask], y.loc[mask].astype(int), base_terms

def lr_fit_coef_on_bi(df_bi: pd.DataFrame, target="asthma_flag"):
    X, y, terms = build_lr_X(df_bi, target)
    if X.shape[0] < 10 or X.shape[1] == 0:
        return None, None, None, None
    Xc = sm.add_constant(X, has_constant="add").astype("float64")
    model = sm.GLM(y.astype("float64"), Xc, family=sm.families.Binomial())
    res = model.fit()
    coef = res.params.drop(labels=["const"])
    return res, coef, terms, set(NUM_COLS) & set(terms)

def anonymize(bi: pd.DataFrame, mode="utility",
              group_key=("AGEBIN","asthma_flag","GENDER","RACE","ETHNICITY"),
              k_min=40, seed=7, logit_align=False):
    rng = np.random.default_rng(seed)
    df = bi.copy()
    for c in NUM_COLS + FLAG_COLS + CAT_COLS:
        if c not in df: df[c] = np.nan
    def coarsen(df0):
        out = df0.copy()
        out["GENDER"] = out["GENDER"].astype(str).str.upper().str[0].map({"M":"M","F":"F"}).fillna("U")
        for c in ["RACE","ETHNICITY"]:
            ser = out[c].astype(str).replace({"":"(NA)"})
            vc = ser.value_counts(dropna=False)
            rare = set(vc[vc < 50].index)
            out[c] = ser.apply(lambda v: "other" if v in rare else v)
        return out
    df = coarsen(df)
    df["AGEBIN"] = agebin(df["AGE"])
    def relax(keys):
        tiers = [
            list(keys),
            [k for k in keys if k != "ETHNICITY"],
            [k for k in keys if k != "RACE"],
            ["AGEBIN","asthma_flag"]
        ]
        for ks in tiers:
            sizes = df.groupby(ks, sort=False).size()
            if (sizes >= k_min).all():
                return ks
        return ["AGEBIN","asthma_flag"]
    key = relax(list(group_key))
    groups = df.groupby(key, sort=False)

    lr_res, lr_coef, lr_terms, cont_terms = lr_fit_coef_on_bi(df, target="asthma_flag")
    w_cont = None
    if logit_align and lr_coef is not None and cont_terms:
        cont_cols = [c for c in ["mean_systolic_bp","mean_diastolic_bp","mean_bmi","mean_weight"] if c in lr_terms]
        if cont_cols:
            w_cont = lr_coef.reindex(cont_cols).to_numpy()
            if not np.all(np.isfinite(w_cont)) or np.linalg.norm(w_cont) < 1e-8:
                w_cont = None

    out = df.copy()
    for gkey, idx in groups.indices.items():
        idl = list(idx)
        sub = df.loc[idl]
        Xref = sub[NUM_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()
        Up = gaussian_copula_group(Xref, rng)
        int_mask = np.array([c in INT_COLS for c in NUM_COLS], dtype=bool)
        Xsyn = quantile_project_uniforms(Up, Xref, int_mask, rng)
        out.loc[idl, NUM_COLS] = Xsyn

    for c in INT_COLS:
        ser = pd.to_numeric(out[c], errors="coerce").round()
        if c == "AGE":
            ser = ser.clip(lower=0, upper=200)
        else:
            ser = ser.clip(lower=0)
        out[c] = ser.fillna(0).astype("int64")
    for c in [col for col in NUM_COLS if col not in INT_COLS]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(3)

    if logit_align and w_cont is not None:
        for yv in [0,1]:
            for ab in DEFAULT_LABELS:
                mask = (df["asthma_flag"]==yv) & (agebin(df["AGE"])==ab)
                mask = mask.fillna(False)
                ids = out.index[mask]
                if len(ids)==0: continue
                def build_X_sub(dsub):
                    Xr, yr, terms = build_lr_X(dsub, "asthma_flag")
                    Xc = sm.add_constant(Xr.reindex(columns=terms), has_constant="add").astype("float64")
                    return Xc, terms, Xr.index
                bi_sub = df.loc[ids]
                ci_sub = out.loc[ids]
                Xc_bi, terms, idx_bi = build_X_sub(bi_sub)
                Xc_ci, terms2, idx_ci = build_X_sub(ci_sub)
                if Xc_bi.shape[0]==0 or Xc_ci.shape[0]==0:
                    continue
                logit_bi = lr_res.predict(Xc_bi, linear=True)
                logit_ci = lr_res.predict(Xc_ci, linear=True)
                r = pd.Series(logit_ci).rank(method="average").to_numpy()
                r = (r-1)/max(len(logit_ci)-1,1)
                xs = np.sort(logit_bi)
                pos = r*(len(xs)-1)
                lo = np.floor(pos).astype(int); hi = np.ceil(pos).astype(int); frac = pos-lo
                target_logit = xs[lo]*(1-frac) + xs[hi]*frac
                delta = target_logit - logit_ci
                cont_cols = [c for c in ["mean_systolic_bp","mean_diastolic_bp","mean_bmi","mean_weight"] if c in terms]
                if not cont_cols:
                    continue
                term_pos = {t:i for i,t in enumerate(terms)}
                w_vec = np.array([lr_coef.get(c, 0.0) for c in cont_cols], dtype=float)
                norm2 = float(np.dot(w_vec, w_vec))
                if not np.isfinite(norm2) or norm2 < 1e-10:
                    continue
                subX = out.loc[idx_ci, cont_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
                adjust = (delta / norm2)[:, None] * w_vec[None, :]
                subX2 = subX + adjust
                for j, c in enumerate(cont_cols):
                    col = subX2[:, j]
                    ref = pd.to_numeric(bi_sub[c], errors="coerce")
                    mn, mx = np.nanpercentile(ref, 0.5), np.nanpercentile(ref, 99.5)
                    out.loc[idx_ci, c] = np.clip(col, mn, mx)
        for c in [col for col in NUM_COLS if col not in INT_COLS]:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(3)

    out = out.drop(columns=["AGEBIN"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi", required=True, help="path to Bi.csv")
    ap.add_argument("--out", required=True, help="output Ci.csv")
    ap.add_argument("--mode", choices=["utility","privacy"], default="utility")
    ap.add_argument("--group", default="AGEBIN,asthma_flag,GENDER,RACE,ETHNICITY")
    ap.add_argument("--kmin", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--logit-align", action="store_true", help="align Ci logits to Bi along continuous-feature direction")
    args = ap.parse_args()

    df = pd.read_csv(args.bi, dtype=str, keep_default_na=False)
    for c in NUM_COLS + FLAG_COLS:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    out = anonymize(df, mode=args.mode,
                    group_key=tuple([s.strip() for s in args.group.split(",") if s.strip()]),
                    k_min=args.kmin, seed=args.seed, logit_align=args.logit_align)
    out.to_csv(args.out, index=False)
    print(f"[OK] Saved to {args.out}")
if __name__ == "__main__":
    main()
