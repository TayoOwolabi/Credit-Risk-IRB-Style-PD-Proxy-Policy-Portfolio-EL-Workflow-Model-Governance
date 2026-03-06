from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def calculate_psi(expected, actual, bins: int = 10) -> float:
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    qs = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(expected.quantile(qs).values)
    if len(breakpoints) < 3:
        return 0.0
    expected_counts = np.histogram(expected, breakpoints)[0]
    actual_counts = np.histogram(actual, breakpoints)[0]
    expected_perc = expected_counts / max(1, len(expected))
    actual_perc = actual_counts / max(1, len(actual))
    psi = np.sum((actual_perc - expected_perc) * np.log((actual_perc + 1e-6) / (expected_perc + 1e-6)))
    return float(psi)

def backtesting_deciles(p, y, n: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": p, "y": y})
    df["decile"] = pd.qcut(df["p"], n, duplicates="drop")
    return df.groupby("decile").agg(
        predicted_rate=("p","mean"),
        actual_rate=("y","mean"),
        count=("y","count")
    ).reset_index()

def cutoff_sensitivity(prob_def, grid=None) -> pd.DataFrame:
    if grid is None:
        grid = np.arange(0.05, 0.55, 0.05)
    rows = []
    prob_def = np.asarray(prob_def)
    for c in grid:
        approve = prob_def <= c
        rows.append({
            "PD_cutoff": float(c),
            "approval_rate": float(approve.mean()),
            "avg_PD_approved": float(prob_def[approve].mean()) if approve.any() else float("nan")
        })
    return pd.DataFrame(rows)

def challenger_random_forest_auc(X_train, y_train, X_test, y_test) -> float:
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    p = rf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, p))
