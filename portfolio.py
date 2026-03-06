from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict

def add_ead_lgd_el(df: pd.DataFrame, approval_col: str = "loan_approval", pd_col: str = "prob_def_proxy",
                   seed: int = 42) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    out["loan_amount_approved"] = np.where(out[approval_col] == 1, rng.integers(5000, 45001, size=len(out)), 0)
    out["EAD"] = out["loan_amount_approved"].astype(float)

    avg_cc = out["Avg_CC_score"].astype(float).values if "Avg_CC_score" in out.columns else np.zeros(len(out))
    lgd = 0.55 - 0.02 * avg_cc
    if "Securities_Account" in out.columns:
        lgd = lgd - 0.05 * out["Securities_Account"].values
    if "CD_Account" in out.columns:
        lgd = lgd - 0.05 * out["CD_Account"].values
    if "Mortgage" in out.columns:
        lgd = lgd + 0.03 * (out["Mortgage"].values > 0).astype(int)
    lgd = np.clip(lgd, 0.25, 0.65)
    out["LGD"] = lgd
    out["EL"] = out[pd_col].values * out["LGD"].values * out["EAD"].values
    return out

def portfolio_summary(df: pd.DataFrame) -> Dict[str, float]:
    total_ead = float(df["EAD"].sum()) if "EAD" in df.columns else 0.0
    total_el = float(df["EL"].sum()) if "EL" in df.columns else 0.0
    approved = int(df["loan_approval"].sum()) if "loan_approval" in df.columns else 0
    return {
        "n_applicants": int(len(df)),
        "approved_count": approved,
        "approval_rate": float(df["loan_approval"].mean()) if "loan_approval" in df.columns else float("nan"),
        "total_EAD": total_ead,
        "portfolio_EL": total_el,
        "EL_rate_on_EAD": (total_el / total_ead) if total_ead > 0 else float("nan"),
        "avg_LGD_approved": float(df.loc[df["loan_approval"]==1, "LGD"].mean()) if approved and "LGD" in df.columns else float("nan"),
    }
