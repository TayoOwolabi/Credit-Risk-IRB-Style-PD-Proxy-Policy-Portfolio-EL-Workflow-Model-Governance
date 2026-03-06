from __future__ import annotations
import pandas as pd
from typing import List
from .scorecard_woe import ScorecardArtifacts

FRIENDLY = {
    "Income": "Low Income",
    "Avg_CC_score": "Low Credit Card Score",
    "Mortgage": "High Mortgage/Leverage",
    "Family": "Higher Dependents",
    "Experience": "Limited Work Experience",
    "Age": "Age Risk",
    "Education": "Lower Education Level",
    "Securities_Account": "No Securities Relationship",
    "CD_Account": "No CD Relationship",
    "Online": "No Online Relationship",
    "CreditCard": "No Credit Card Relationship",
}

def add_reason_codes(scored: pd.DataFrame, X_woe: pd.DataFrame, art: ScorecardArtifacts,
                     top_n: int = 3, decline_only: bool = True) -> pd.DataFrame:
    out = scored.copy()
    contrib = pd.DataFrame(index=out.index)
    for f in art.features:
        contrib[f] = -art.factor * art.coefficients[f] * X_woe[f]

    def top_reasons(row: pd.Series) -> List[str]:
        worst = row.sort_values().head(top_n).index.tolist()
        return [FRIENDLY.get(f, f) for f in worst]

    declined_mask = (out["loan_approval"] == 0) if decline_only and "loan_approval" in out.columns else pd.Series([True]*len(out), index=out.index)

    r1, r2, r3 = [], [], []
    for idx in out.index:
        if declined_mask.loc[idx]:
            rs = top_reasons(contrib.loc[idx])
            rs = (rs + ["", "", ""])[:3]
        else:
            rs = ["", "", ""]
        r1.append(rs[0]); r2.append(rs[1]); r3.append(rs[2])

    out["reason_1"] = r1
    out["reason_2"] = r2
    out["reason_3"] = r3
    out["reason_codes"] = out[["reason_1","reason_2","reason_3"]].fillna("").agg(" | ".join, axis=1).str.strip(" |")
    return out
