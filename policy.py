from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def assign_score_band(score: float) -> str:
    if score >= 760: return "A"
    if score >= 700: return "B"
    if score >= 640: return "C"
    if score >= 580: return "D"
    return "E"

def add_bands(df: pd.DataFrame, score_col: str = "credit_score_300_850") -> pd.DataFrame:
    out = df.copy()
    out["score_band"] = out[score_col].apply(assign_score_band)
    return out

def add_pricing(df: pd.DataFrame, pd_col: str = "prob_def_proxy") -> pd.DataFrame:
    out = df.copy()
    APR_MIN, APR_MAX = 0.08, 0.32
    pd_scaled = np.clip(out[pd_col].values, 0.01, 0.50)
    apr_risk = APR_MIN + (APR_MAX - APR_MIN) * (pd_scaled / 0.50)

    band_targets: Dict[str, Tuple[float, float]] = {
        "A": (0.08, 0.12),
        "B": (0.11, 0.16),
        "C": (0.15, 0.22),
        "D": (0.20, 0.28),
        "E": (0.26, 0.32),
    }

    def clip_band(band: str, apr: float) -> float:
        lo, hi = band_targets[band]
        return float(np.clip(apr, lo, hi))

    out["apr_recommended"] = [clip_band(b, a) for b, a in zip(out["score_band"].values, apr_risk)]

    def tier(apr: float) -> str:
        if apr <= 0.12: return "Prime"
        if apr <= 0.16: return "Near-prime"
        if apr <= 0.22: return "Standard"
        if apr <= 0.28: return "Elevated"
        return "High-risk"

    out["pricing_tier"] = out["apr_recommended"].apply(tier)
    return out
