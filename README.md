# FundiFix Predictive Maintenance

Predictive maintenance analytics for rural water infrastructure in Kenya, built for FundiFix Ltd.

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
