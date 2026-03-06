from __future__ import annotations
import argparse, os
from irb_scorecard.io import load_csv
from irb_scorecard.pd_proxy import train_pd_proxy, save_pd_proxy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_csv(args.infile)

    art = train_pd_proxy(df)
    save_pd_proxy(art, os.path.join(args.outdir, "pd_proxy_model_calibrated.joblib"))
    art.metrics.to_csv(os.path.join(args.outdir, "pd_proxy_metrics.csv"), index=False)
    print("Saved PD proxy model + metrics to:", args.outdir)

if __name__ == "__main__":
    main()
