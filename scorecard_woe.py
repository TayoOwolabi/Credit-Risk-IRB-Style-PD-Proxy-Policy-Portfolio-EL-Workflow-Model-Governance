from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

@dataclass
class BinDef:
    edges: Optional[List[float]] = None
    categories: Optional[List[str]] = None

@dataclass
class WoeMapping:
    woe_by_bin: Dict[str, float]
    counts_by_bin: Dict[str, Tuple[int, int]]

@dataclass
class ScorecardArtifacts:
    feature_bins: Dict[str, BinDef]
    feature_woe: Dict[str, WoeMapping]
    coefficients: Dict[str, float]
    intercept: float
    features: List[str]
    numeric_features: List[str]
    binary_features: List[str]
    pdo: float
    base_score: float
    base_odds_good: float
    score_min: int
    score_max: int
    factor: float
    offset: float
    base_points: float
    pd_cutoff: float
    model: Any

def _infer_target(df: pd.DataFrame) -> str:
    if "Personal_loan" in df.columns:
        return "Personal_loan"
    if "Personal.loan" in df.columns:
        return "Personal.loan"
    raise ValueError("Target not found: Personal_loan / Personal.loan")

def make_numeric_bins(series: pd.Series, n_bins: int = 10) -> BinDef:
    s = series.dropna()
    uniq = s.nunique()
    bins = min(n_bins, max(2, uniq))
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(s.quantile(qs).values)
    if len(edges) < 3:
        edges = np.unique([s.min(), s.median(), s.max()])
    edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
    return BinDef(edges=list(map(float, edges)))

def assign_numeric_bin(x: float, edges: List[float]) -> str:
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = int(np.clip(idx, 0, len(edges) - 2))
    lo = edges[idx]; hi = edges[idx + 1]
    return f"({lo:.6g},{hi:.6g}]"

def compute_woe(x_bins: pd.Series, y_bad: pd.Series, smoothing: float = 0.5) -> WoeMapping:
    tmp = pd.DataFrame({"bin": x_bins.astype(str), "bad": y_bad.values})
    g = tmp.groupby("bin")["bad"]
    bad_count = g.sum()
    total = g.count()
    good_count = total - bad_count

    bad_count_s = bad_count + smoothing
    good_count_s = good_count + smoothing
    bad_total_s = bad_count_s.sum()
    good_total_s = good_count_s.sum()

    bad_dist = bad_count_s / bad_total_s
    good_dist = good_count_s / good_total_s
    woe = np.log(good_dist / bad_dist)

    woe_by_bin = woe.to_dict()
    counts_by_bin = {k: (int(good_count[k]), int(bad_count[k])) for k in total.index}
    return WoeMapping(woe_by_bin=woe_by_bin, counts_by_bin=counts_by_bin)

def fit_woe(X_dev: pd.DataFrame, y_bad: pd.Series, numeric_cols: List[str], binary_cols: List[str],
            n_bins: int = 10, smoothing: float = 0.5):
    feature_bins: Dict[str, BinDef] = {}
    feature_woe: Dict[str, WoeMapping] = {}
    X_woe = pd.DataFrame(index=X_dev.index)

    for col in numeric_cols:
        bdef = make_numeric_bins(X_dev[col], n_bins=n_bins)
        feature_bins[col] = bdef
        bins_dev = X_dev[col].apply(lambda v: "MISSING" if pd.isna(v) else assign_numeric_bin(float(v), bdef.edges))
        wmap = compute_woe(bins_dev, y_bad, smoothing=smoothing)
        if "MISSING" not in wmap.woe_by_bin:
            wmap.woe_by_bin["MISSING"] = 0.0
            wmap.counts_by_bin["MISSING"] = (0, 0)
        feature_woe[col] = wmap
        X_woe[col] = bins_dev.map(lambda b: float(wmap.woe_by_bin.get(b, 0.0)))

    for col in binary_cols:
        bdef = BinDef(categories=sorted(X_dev[col].dropna().astype(str).unique().tolist()))
        feature_bins[col] = bdef
        bins_dev = X_dev[col].apply(lambda v: "MISSING" if pd.isna(v) else str(v))
        wmap = compute_woe(bins_dev, y_bad, smoothing=smoothing)
        if "MISSING" not in wmap.woe_by_bin:
            wmap.woe_by_bin["MISSING"] = 0.0
            wmap.counts_by_bin["MISSING"] = (0, 0)
        feature_woe[col] = wmap
        X_woe[col] = bins_dev.map(lambda b: float(wmap.woe_by_bin.get(b, 0.0)))

    return X_woe, feature_bins, feature_woe

def transform_with_woe(X_any: pd.DataFrame, feature_bins: Dict[str, BinDef], feature_woe: Dict[str, WoeMapping],
                       features: List[str]) -> pd.DataFrame:
    Xw = pd.DataFrame(index=X_any.index)
    for f in features:
        bdef = feature_bins[f]; wmap = feature_woe[f]
        if bdef.edges is not None:
            bins = X_any[f].apply(lambda v: "MISSING" if pd.isna(v) else assign_numeric_bin(float(v), bdef.edges)).astype(str)
        else:
            bins = X_any[f].apply(lambda v: "MISSING" if pd.isna(v) else str(v)).astype(str)
        Xw[f] = bins.map(lambda b: float(wmap.woe_by_bin.get(b, 0.0)))
    return Xw

def train_scorecard(df: pd.DataFrame, pdo: float = 50, base_score: float = 600, base_odds_good: float = 20,
                    score_min: int = 300, score_max: int = 850, random_state: int = 42) -> ScorecardArtifacts:
    target = _infer_target(df)
    y_approve = df[target].astype(int)
    y_bad = (1 - y_approve).astype(int)

    drop_cols = [c for c in ["ID","ZIP_Code"] if c in df.columns]
    X = df.drop(columns=[target] + drop_cols, errors="ignore")

    numeric_features = [c for c in ["Age","Experience","Income","Family","Avg_CC_score","Education","Mortgage"] if c in X.columns]
    binary_features = [c for c in ["Securities_Account","CD_Account","Online","CreditCard"] if c in X.columns]
    features = numeric_features + binary_features
    X = X[features].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_bad, test_size=0.40, random_state=random_state, stratify=y_bad
    )
    X_val, _, y_val, _ = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    X_train_woe, feature_bins, feature_woe = fit_woe(X_train, y_train, numeric_features, binary_features, n_bins=10, smoothing=0.5)
    X_val_woe = transform_with_woe(X_val, feature_bins, feature_woe, features)

    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train_woe, y_train)

    p_bad_val = model.predict_proba(X_val_woe)[:, 1]
    target_approval_rate = float(y_approve.mean())
    grid = np.linspace(0.01, 0.50, 200)
    best_cut, best_gap = None, 1e9
    for c in grid:
        appr = float((p_bad_val <= c).mean())
        gap = abs(appr - target_approval_rate)
        if gap < best_gap:
            best_gap = gap
            best_cut = c
    pd_cutoff = float(best_cut)

    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds_good)
    coef = dict(zip(features, model.coef_[0].tolist()))
    intercept = float(model.intercept_[0])
    base_points = float(offset - factor * intercept)

    return ScorecardArtifacts(
        feature_bins=feature_bins,
        feature_woe=feature_woe,
        coefficients={k: float(v) for k, v in coef.items()},
        intercept=intercept,
        features=features,
        numeric_features=numeric_features,
        binary_features=binary_features,
        pdo=float(pdo),
        base_score=float(base_score),
        base_odds_good=float(base_odds_good),
        score_min=int(score_min),
        score_max=int(score_max),
        factor=float(factor),
        offset=float(offset),
        base_points=float(base_points),
        pd_cutoff=pd_cutoff,
        model=model,
    )

def predict_pbad_and_score(df: pd.DataFrame, art: ScorecardArtifacts):
    target = _infer_target(df)
    drop_cols = [c for c in ["ID","ZIP_Code"] if c in df.columns]
    X = df.drop(columns=[target] + drop_cols, errors="ignore")[art.features].copy()
    Xw = transform_with_woe(X, art.feature_bins, art.feature_woe, art.features)

    p_bad = art.model.predict_proba(Xw)[:, 1]
    p_bad = np.clip(p_bad, 1e-6, 1 - 1e-6)
    odds_good = (1 - p_bad) / p_bad

    score = art.offset + art.factor * np.log(odds_good)
    score = np.clip(score, art.score_min, art.score_max)

    out = df.copy()
    out["prob_def_proxy"] = p_bad
    out["credit_score_300_850"] = score
    out["loan_approval"] = (p_bad <= art.pd_cutoff).astype(int)
    return out, Xw

def build_points_table(art: ScorecardArtifacts) -> pd.DataFrame:
    rows = []
    for f in art.features:
        wmap = art.feature_woe[f]
        for bin_label, woe_val in wmap.woe_by_bin.items():
            pts = -art.factor * art.coefficients[f] * float(woe_val)
            good_cnt, bad_cnt = wmap.counts_by_bin.get(bin_label, (0, 0))
            rows.append({
                "feature": f,
                "bin": bin_label,
                "woe": float(woe_val),
                "coef": float(art.coefficients[f]),
                "points": float(pts),
                "good_count_dev": int(good_cnt),
                "bad_count_dev": int(bad_cnt),
            })
    return pd.DataFrame(rows).sort_values(["feature","bin"]).reset_index(drop=True)

def evaluate_scorecard(df: pd.DataFrame, art: ScorecardArtifacts) -> pd.DataFrame:
    target = _infer_target(df)
    y_bad = (1 - df[target].astype(int)).astype(int)
    drop_cols = [c for c in ["ID","ZIP_Code"] if c in df.columns]
    X = df.drop(columns=[target] + drop_cols, errors="ignore")[art.features].copy()
    Xw = transform_with_woe(X, art.feature_bins, art.feature_woe, art.features)
    p = art.model.predict_proba(Xw)[:, 1]
    return pd.DataFrame({
        "AUC_bad": [roc_auc_score(y_bad, p)],
        "Brier_bad": [brier_score_loss(y_bad, p)],
        "avg_pred_bad": [float(p.mean())],
        "actual_bad_rate": [float(y_bad.mean())],
    })

def save_scorecard(art: ScorecardArtifacts, path: str) -> None:
    joblib.dump(art, path)

def load_scorecard(path: str) -> ScorecardArtifacts:
    return joblib.load(path)
