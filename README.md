# (1) Summary

Financial institutions must continuously evaluate borrower risk while maintaining portfolio stability and regulatory compliance.

This project builds a credit risk decision framework that:

• Estimates borrower Probability of Default (PD)
• Applies credit policy thresholds for lending decisions
• Simulates portfolio expected losses
• Evaluates risk exposure under varying economic conditions

The model demonstrates how predictive analytics and financial modeling can support risk-aware lending strategies, portfolio monitoring, and capital planning.

# (2) Business Problem

Banks and lending institutions face a core challenge:

How can credit decisions be optimized while controlling portfolio risk and expected losses?

Traditional credit approval methods often rely on static borrower metrics that may not fully capture changing risk conditions.

Key challenges include:

• Identifying high-risk borrowers before loan approval
• Estimating expected credit losses across a loan portfolio
• Aligning lending policies with institutional risk appetite
• Monitoring portfolio risk exposure under economic stress scenarios

This project addresses these challenges by developing a data-driven credit risk modeling framework.

# (3) Data & Methodology

The modeling approach follows standard credit risk analytics methodology.

# Data Inputs

Borrower features used in the model include:

• income
• debt-to-income ratio
• credit history indicators
• loan amount
• repayment history

# Modeling Techniques

# The framework uses:

• Weight of Evidence (WoE) feature transformation
• Logistic regression for PD estimation
• Portfolio expected loss simulation
• Risk segmentation and borrower classification

These techniques reflect methodologies commonly used in retail credit risk modeling and Basel internal rating systems.

# (4) Financial Model

The credit risk model estimates the Probability of Default (PD) for each borrower.

The expected loss framework follows standard financial risk modeling principles.

Expected Loss (EL) = PD X LGD X EAD
         
Where:

PD = Probability of Default
LGD = Loss Given Default
EAD = Exposure at Default

The model calculates expected losses across the loan portfolio, enabling financial institutions to evaluate potential risk exposure.

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
