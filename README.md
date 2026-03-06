# Credit Risk PD Proxy + IRB-Style Policy + Traditional WoE Scorecard (PDO 300–850)

This repository is a **GitHub-ready credit risk portfolio project** that demonstrates:

- **IRB-style development workflow** (DEV/VAL/TEST split, calibration, policy layer)
- **PD proxy modeling** (because the dataset target is *loan approval*, not default)
- **Portfolio risk metrics** (EAD / LGD / EL)
- **Model governance** (PSI, backtesting tables, challenger model, cutoff sensitivity)
- **Traditional bank scorecard**: **Binning → WoE → Logistic Regression → PDO score scaling (300–850)**
- **Score bands (A/B/C/D/E)**, **pricing tiers**, and **reason codes** (Top-3 drivers) for declines

> ⚠️ Important: The dataset target `Personal.loan` is **loan approval** (1=approved, 0=not approved).  
> A true Basel/IRB PD model requires a **default label** (1=default/bad).  
> In this project, we compute a **PD proxy**: `prob_def_proxy = 1 - P(approved)`.

---

## Repo Structure

```
credit-risk-pd-scorecard-woe/
├── data/
│   └── Bank_Loan_Approval_copy.csv
├── notebooks/
│   └── Credit_Risk_Benchmark_Validated.ipynb
├── src/
│   └── irb_scorecard/
│       ├── __init__.py
│       ├── io.py
│       ├── pd_proxy.py
│       ├── scorecard_woe.py
│       ├── governance.py
│       ├── policy.py
│       ├── portfolio.py
│       └── reason_codes.py
├── scripts/
│   ├── train_pd_proxy.py
│   ├── train_scorecard.py
│   ├── score_dataset.py
│   ├── run_governance.py
│   └── make_report.py
├── artifacts/               # generated
├── outputs/                 # generated
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── .gitignore
```

---

## Quickstart

### 1) Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Train the IRB-style PD proxy model (calibrated)
```bash
python scripts/train_pd_proxy.py --infile data/Bank_Loan_Approval_copy.csv --outdir artifacts
```

### 3) Train the traditional WoE scorecard + PDO score scaling (300–850)
```bash
python scripts/train_scorecard.py --infile data/Bank_Loan_Approval_copy.csv --outdir artifacts
```

### 4) Score a dataset (adds prob_def + approvals + score bands + pricing + reason codes)
```bash
python scripts/score_dataset.py --infile data/Bank_Loan_Approval_copy.csv --outdir artifacts
```

### 5) Run governance checks (PSI, backtesting, challenger, cutoff sensitivity)
```bash
python scripts/run_governance.py --infile data/Bank_Loan_Approval_copy.csv --artifacts artifacts --outdir outputs
```

### 6) Generate a Markdown report
```bash
python scripts/make_report.py --outputs outputs --artifacts artifacts --out outputs/REPORT.md
```

---

## Key Outputs (generated)

- `artifacts/pd_proxy_model_calibrated.joblib`
- `artifacts/scorecard_model_woe.joblib`
- `artifacts/scored_pd_proxy.csv`
- `artifacts/scored_scorecard.csv`
- `artifacts/scorecard_points_table.csv`
- `artifacts/portfolio_summary.json`
- `outputs/governance_summary.json`
- `outputs/backtesting_deciles.csv`
- `outputs/cutoff_sensitivity.csv`

---

## License
MIT License — see `LICENSE`.
