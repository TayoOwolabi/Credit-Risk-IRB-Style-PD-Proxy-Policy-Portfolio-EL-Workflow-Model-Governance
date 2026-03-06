from __future__ import annotations
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_").replace(".", "_") for c in df.columns]
    df.rename(columns={"Avg_CC_Score": "Avg_CC_score"}, inplace=True)
    return df
