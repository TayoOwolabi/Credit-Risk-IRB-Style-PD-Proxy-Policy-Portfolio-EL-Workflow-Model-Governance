from __future__ import annotations
import argparse, os
from irb_scorecard.io import load_csv
from irb_scorecard.scorecard_woe import train_scorecard, save_scorecard, build_points_table, evaluate_scorecard

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pdo", type=float, default=50)
    ap.add_argument("--base-score", type=float, default=600)
    ap.add_argument("--base-odds-good", type=float, default=20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_csv(args.infile)

    art = train_scorecard(df, pdo=args.pdo, base_score=args.base_score, base_odds_good=args.base_odds_good)
    save_scorecard(art, os.path.join(args.outdir, "scorecard_model_woe.joblib"))
    build_points_table(art).to_csv(os.path.join(args.outdir, "scorecard_points_table.csv"), index=False)
    evaluate_scorecard(df, art).to_csv(os.path.join(args.outdir, "scorecard_metrics.csv"), index=False)

    print("Saved scorecard model + points table + metrics to:", args.outdir)

if __name__ == "__main__":
    main()
