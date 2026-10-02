# FundiFix Predictive Maintenance

Predictive maintenance analytics for rural water infrastructure in Kenya, built for [FundiFix Ltd](https://fundifix.org/).

## Problem

FundiFix operates a subscription-based, guaranteed-repair service for rural water points in Kwale and Kitui counties. Technicians are currently dispatched reactively. This project builds a predictive model to identify water points at elevated risk of failure before a fault is reported, enabling proactive maintenance scheduling.

## Dataset

[Water Point Data Exchange (WPdx) — Kenya](https://data.humdata.org/dataset/wpdx_ken), from OCHA HDX. 21,953 water point records, 54 fields. Target variable: `status_clean`, collapsed to Functional / Needs Repair / Non-Functional.

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| Modeling | scikit-learn, XGBoost |
| Explainability | SHAP |
| Data storage | SQLite / DuckDB |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version control | GitHub (Free plan) |
| Data versioning | DVC |

All components are free and require no billing account.

## Setup

```bash
pip install -r requirements.txt
```

## License

Apache 2.0

## Module 4: model, explanations, fairness and API

Run from the repo root, in this order, after the Module 3 pipeline (`python src/data/pipeline.py`):

```bash
python -m src.models.train         # tune 3 models, log to MLflow, register best, save models/fundifix_model.joblib
python -m src.models.explain       # SHAP global and local plots, DiCE counterfactuals
python -m src.models.fairness      # Fairlearn metrics by county, urban/rural, road access, source + mitigation
python -m src.models.sensitivity   # robustness to input changes
pytest tests -v                    # pipeline and API tests
mlflow ui --backend-store-uri sqlite:///mlflow.db    # view experiments at http://127.0.0.1:5000
uvicorn api.main:app --reload      # API docs at http://127.0.0.1:8000/docs
```

Outputs: `reports/figures/`, `reports/metrics/`, `docs/model_card.md`, `docs/fairness_report.md`, `notebooks/04_evaluation.ipynb`.
