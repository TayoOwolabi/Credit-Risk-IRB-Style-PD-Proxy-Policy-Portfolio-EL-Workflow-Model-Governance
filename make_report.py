from __future__ import annotations
import argparse, os, json
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gov_path = os.path.join(args.outputs, "governance_summary.json")
    port_path = os.path.join(args.artifacts, "portfolio_summary.json")

    pd_metrics = os.path.join(args.artifacts, "pd_proxy_metrics.csv")
    sc_metrics = os.path.join(args.artifacts, "scorecard_metrics.csv")

    gov = json.load(open(gov_path, "r", encoding="utf-8")) if os.path.exists(gov_path) else {}
    port = json.load(open(port_path, "r", encoding="utf-8")) if os.path.exists(port_path) else {}
    pm = pd.read_csv(pd_metrics) if os.path.exists(pd_metrics) else pd.DataFrame()
    sm = pd.read_csv(sc_metrics) if os.path.exists(sc_metrics) else pd.DataFrame()

    lines = []
    lines += ["# Credit Risk Project Report", ""]
    lines += ["## PD Proxy Model Metrics", ""]
    lines += [pm.to_markdown(index=False) if not pm.empty else "_Not found._", ""]
    lines += ["## Scorecard Metrics", ""]
    lines += [sm.to_markdown(index=False) if not sm.empty else "_Not found._", ""]
    lines += ["## Governance Summary", ""]
    if gov:
        lines += [f"- PSI (train vs test PD proxy): **{gov.get('psi_pd_train_vs_test','n/a')}**"]
        lines += [f"- RF challenger AUC (approval): **{gov.get('random_forest_challenger_auc_on_approval','n/a')}**"]
    else:
        lines += ["_Not found._"]
    lines += ["", "## Portfolio Summary (Scorecard-based)", ""]
    if port:
        for k,v in port.items():
            lines += [f"- **{k}**: {v}"]
    else:
        lines += ["_Not found._"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Wrote report to:", args.out)

if __name__ == "__main__":
    main()
