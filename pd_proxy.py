from __future__ import annotations
import numpy as np
import pandas as pd
import joblib

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

class Winsorizer(BaseEstimator, TransformerMixin):
    """Caps numeric outliers using training quantiles (fit on DEV only)."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
        self.bounds_: Dict[str, Tuple[float, float]] = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for c in X.columns:
            lo = X[c].quantile(self.lower)
            hi = X[c].quantile(self.upper)
            self.bounds_[c] = (float(lo), float(hi))
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c, (lo, hi) in self.bounds_.items():
            X[c] = X[c].clip(lo, hi)
        return X

@dataclass
class PDProxyArtifacts:
    model: Any
    metrics: pd.DataFrame
    pd_cutoff: float
    feature_cols: List[str]
    numeric_cols: List[str]
    binary_cols: List[str]

def _infer_target(df: pd.DataFrame) -> str:
    if "Personal_loan" in df.columns:
        return "Personal_loan"
    if "Personal.loan" in df.columns:
        return "Personal.loan"
    raise ValueError("Target not found: Personal_loan / Personal.loan")

def train_pd_proxy(df: pd.DataFrame, random_state: int = 42) -> PDProxyArtifacts:
    target = _infer_target(df)
    y = df[target].astype(int)

    drop_cols = [c for c in ["ID", "ZIP_Code"] if c in df.columns]
    X = df.drop(columns=[target] + drop_cols, errors="ignore")

    numeric_cols = [c for c in ["Age","Experience","Income","Family","Avg_CC_score","Education","Mortgage"] if c in X.columns]
    binary_cols = [c for c in ["Securities_Account","CD_Account","Online","CreditCard"] if c in X.columns]
    feature_cols = numeric_cols + binary_cols
    X = X[feature_cols].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("winsor", Winsorizer(0.01, 0.99)),
        ("scaler", StandardScaler()),
    ])
    bin_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_cols),
            ("bin", bin_tf, binary_cols),
        ],
        remainder="drop"
    )

    base_logit = LogisticRegression(
        max_iter=2000, solver="lbfgs", class_weight="balanced"
    )
    pipe = Pipeline(steps=[("preprocess", preprocess), ("logit", base_logit)])

    calibrated = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)

    def eval_split(Xs, ys, name):
        p = calibrated.predict_proba(Xs)[:, 1]
        return {
            "dataset": name,
            "AUC": roc_auc_score(ys, p),
            "Brier": brier_score_loss(ys, p),
            "predicted_approval_rate_at_0_5": float((p >= 0.5).mean()),
            "actual_approval_rate": float(ys.mean())
        }

    metrics = pd.DataFrame([
        eval_split(X_train, y_train, "train"),
        eval_split(X_val, y_val, "val"),
        eval_split(X_test, y_test, "test"),
    ])

    p_val = calibrated.predict_proba(X_val)[:, 1]
    pd_val = 1.0 - p_val
    target_approval_rate = float(y.mean())

    grid = np.linspace(0.05, 0.95, 181)
    best_cut, best_gap = None, 1e9
    for c in grid:
        appr = float((pd_val <= c).mean())
        gap = abs(appr - target_approval_rate)
        if gap < best_gap:
            best_gap = gap
            best_cut = c
    pd_cutoff = float(best_cut)

    return PDProxyArtifacts(
        model=calibrated,
        metrics=metrics,
        pd_cutoff=pd_cutoff,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        binary_cols=binary_cols,
    )

def score_pd_proxy(df: pd.DataFrame, artifacts: PDProxyArtifacts, seed: int = 42) -> pd.DataFrame:
    target = _infer_target(df)
    drop_cols = [c for c in ["ID", "ZIP_Code"] if c in df.columns]
    X = df.drop(columns=[target] + drop_cols, errors="ignore")[artifacts.feature_cols].copy()

    p_approve = artifacts.model.predict_proba(X)[:, 1]
    prob_def = 1.0 - p_approve
    loan_approval = (prob_def <= artifacts.pd_cutoff).astype(int)

    rng = np.random.default_rng(seed)
    loan_amount = np.where(loan_approval == 1, rng.integers(5000, 45001, size=len(df)), 0)
    ead = loan_amount.astype(float)

    avg_cc = df["Avg_CC_score"].astype(float).values if "Avg_CC_score" in df.columns else np.zeros(len(df))
    lgd = 0.55 - 0.02 * avg_cc
    if "Securities_Account" in df.columns:
        lgd = lgd - 0.05 * df["Securities_Account"].values
    if "CD_Account" in df.columns:
        lgd = lgd - 0.05 * df["CD_Account"].values
    if "Mortgage" in df.columns:
        lgd = lgd + 0.03 * (df["Mortgage"].values > 0).astype(int)
    lgd = np.clip(lgd, 0.25, 0.65)

    el = prob_def * lgd * ead

    out = df.copy()
    out["prob_def"] = prob_def
    out["loan_approval"] = loan_approval
    out["loan_amount_approved"] = loan_amount
    out["EAD"] = ead
    out["LGD"] = lgd
    out["EL"] = el
    return out

def save_pd_proxy(artifacts: PDProxyArtifacts, path: str) -> None:
    joblib.dump(artifacts, path)

def load_pd_proxy(path: str) -> PDProxyArtifacts:
    return joblib.load(path)
