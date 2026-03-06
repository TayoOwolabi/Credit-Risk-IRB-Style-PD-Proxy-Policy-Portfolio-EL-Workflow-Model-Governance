from __future__ import annotations
import argparse, os, json
from irb_scorecard.io import load_csv
from irb_scorecard.pd_proxy import load_pd_proxy, score_pd_proxy, train_pd_proxy, save_pd_proxy
from irb_scorecard.scorecard_woe import load_scorecard, predict_pbad_and_score, train_scorecard, save_scorecard, build_points_table
from irb_scorecard.policy import add_bands, add_pricing
from irb_scorecard.portfolio import add_ead_lgd_el, portfolio_summary
from irb_scorecard.reason_codes import add_reason_codes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_csv(args.infile)

    pd_path = os.path.join(args.outdir, "pd_proxy_model_calibrated.joblib")
    if not os.path.exists(pd_path):
        pd_art = train_pd_proxy(df); save_pd_proxy(pd_art, pd_path)
    pd_art = load_pd_proxy(pd_path)

    sc_path = os.path.join(args.outdir, "scorecard_model_woe.joblib")
    if not os.path.exists(sc_path):
        sc_art = train_scorecard(df); save_scorecard(sc_art, sc_path)
        build_points_table(sc_art).to_csv(os.path.join(args.outdir, "scorecard_points_table.csv"), index=False)
    sc_art = load_scorecard(sc_path)

    scored_pd = score_pd_proxy(df, pd_art)
    scored_pd.to_csv(os.path.join(args.outdir, "scored_pd_proxy.csv"), index=False)

    scored_sc, Xw = predict_pbad_and_score(df, sc_art)
    scored_sc = add_bands(scored_sc)
    scored_sc = add_pricing(scored_sc, pd_col="prob_def_proxy")
    scored_sc = add_reason_codes(scored_sc, Xw, sc_art, top_n=3, decline_only=True)
    scored_sc = add_ead_lgd_el(scored_sc, approval_col="loan_approval", pd_col="prob_def_proxy")
    scored_sc.to_csv(os.path.join(args.outdir, "scored_scorecard.csv"), index=False)

    summary = portfolio_summary(scored_sc)
    with open(os.path.join(args.outdir, "portfolio_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved scored outputs to:", args.outdir)

if __name__ == "__main__":
    main()
