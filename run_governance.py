from __future__ import annotations
import argparse, os, json
from sklearn.model_selection import train_test_split
from irb_scorecard.io import load_csv
from irb_scorecard.pd_proxy import train_pd_proxy
from irb_scorecard.governance import calculate_psi, backtesting_deciles, cutoff_sensitivity, challenger_random_forest_auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_csv(args.infile)
    target = "Personal_loan" if "Personal_loan" in df.columns else "Personal.loan"
    y = df[target].astype(int)
    X = df.drop(columns=[target], errors="ignore")

    art = train_pd_proxy(df)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    p_train = art.model.predict_proba(X_train[art.feature_cols])[:, 1]
    p_test = art.model.predict_proba(X_test[art.feature_cols])[:, 1]
    pd_train = 1.0 - p_train
    pd_test = 1.0 - p_test

    psi_pd = calculate_psi(pd_train, pd_test, bins=10)
    bt = backtesting_deciles(p_test, y_test.values, n=10)
    bt.to_csv(os.path.join(args.outdir, "backtesting_deciles.csv"), index=False)

    sens = cutoff_sensitivity(pd_test)
    sens.to_csv(os.path.join(args.outdir, "cutoff_sensitivity.csv"), index=False)

    rf_auc = challenger_random_forest_auc(
        X_train[art.feature_cols], y_train.values,
        X_test[art.feature_cols], y_test.values
    )

    summary = {
        "psi_pd_train_vs_test": psi_pd,
        "random_forest_challenger_auc_on_approval": rf_auc,
        "notes": [
            "Dataset target is approval; governance uses approval propensity and PD proxy (1 - P(approve)).",
            "PSI thresholds: <0.1 stable, 0.1–0.25 moderate, >0.25 major shift."
        ]
    }
    with open(os.path.join(args.outdir, "governance_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved governance outputs to:", args.outdir)

if __name__ == "__main__":
    main()
