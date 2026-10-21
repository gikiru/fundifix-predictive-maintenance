# FundiFix Predictive Maintenance

[![ci-cd](https://github.com/gikiru/fundifix-predictive-maintenance/actions/workflows/ci.yml/badge.svg)](https://github.com/gikiru/fundifix-predictive-maintenance/actions/workflows/ci.yml)

Decision support for [FundiFix Ltd](https://fundifix.org/), which repairs rural water points in Kitui and Kwale, Kenya, under a three-day repair guarantee. The system ranks water points by their chance of being broken now, explains each rating, checks it for fairness, and monitors itself after deployment.

MSc Data Analytics capstone, Nexford University. Modules 1 to 5 and the Final Project.

## Live system

| What | Link |
|---|---|
| Stakeholder dashboard (with explainability and fairness tabs) | `[insert Streamlit Cloud URL]` |
| Monitoring dashboard | same app, **Monitoring** tab |
| Live API (`/predict`, `/explanations/{id}`) | `https://<hf-username>-fundifix-api.hf.space/docs` |
| CI/CD runs | [GitHub Actions](https://github.com/gikiru/fundifix-predictive-maintenance/actions) |

## Results in one paragraph

On 1,955 test water points, a visit list built from the model covers 41% of points and finds 69% of the broken ones, about 27 broken points per 100 visits against 16 for random visits. Balanced accuracy is 0.567, below the Module 1 target of 0.80. 12 of the 16 public survey sources label every point broken, so the model is trained on the 4 sources whose labels vary. It has not been validated in Kitui or Kwale. Full detail: [Model Card](docs/model_card.md), [Fairness report](docs/fairness_report.md).

## Architecture

```
WPdx CSVs ─► Prefect pipeline ─► DuckDB ─► training (MLflow) ─► registered model ─┬─► FastAPI (Hugging Face Space) ─► explanation store
 (data/raw)   ingest, clean,       clean      RF / XGBoost,       models/            │                                        │
              validate (GE),       table      SHAP, LIME, DiCE,   fundifix_model     └─► Streamlit dashboard ◄── monitoring ◄──┘
              bias check, anonymise           Fairlearn                                    (8 tabs)            Evidently + Fairlearn
                                                                                                               weekly, retrain with gates
```

## Repository map

| Path | Contents | Module |
|---|---|---|
| `src/data/` | Prefect pipeline: ingest (with SHA-256 check), clean, transform, validate (Great Expectations), bias check, anonymise, audit log | 3 |
| `src/models/` | training with MLflow, SHAP and DiCE (`explain.py`), LIME cross-check, Fairlearn analysis, sensitivity, retraining with release gates, explanation store | 4, Final |
| `src/monitoring/monitor.py` | data drift, prediction drift, performance and bias drift, with alerts | Final |
| `src/flows/mlops_flow.py` | weekly Prefect flow: monitor, then retrain if new labels or a red alert | Final |
| `api/` | FastAPI service | 4, Final |
| `app/` | Streamlit dashboard and the data it reads (`app/data/monitoring/` holds the latest monitoring run) | 5, Final |
| `docker/`, `Dockerfile`, `docker-compose.yml` | containers for pipeline, API, dashboard, monitoring, MLflow | 3, Final |
| `.github/workflows/` | CI/CD (`ci.yml`) and weekly uptime check (`uptime.yml`) | Final |
| `tests/` | 27 tests: pipeline, API and audit store, monitoring alerts, retraining checks | 3 to Final |
| `docs/` | Model Card, fairness report, data dictionary, RAID log, reconciliation log, Ethical AI framework, impact assessment, compliance report, monitoring plan | 2 to Final |

## Run it

Requires Python 3.12 and the two WPdx files in `data/raw/` (see `data/raw/README.md`).

```bash
pip install -r requirements.txt
python src/data/pipeline.py                 # Module 3 pipeline
python -m src.models.train                  # train, log to MLflow, register champion
python -m src.models.explain                # SHAP and counterfactuals
python -m src.models.lime_check             # LIME cross-check of SHAP reasons
python -m src.models.fairness               # Fairlearn metrics and mitigation
python -m src.models.dashboard_data         # refresh app/data/ for the dashboard
python -m src.monitoring.monitor --set-baseline    # record go-live results, then run monitoring
pytest tests -v
streamlit run app/dashboard.py              # http://localhost:8501
uvicorn api.main:app                        # http://127.0.0.1:8000/docs
mlflow ui --backend-store-uri sqlite:///mlflow.db  # http://127.0.0.1:5000
```

### With Docker Compose

```bash
docker compose up --build          # pipeline runs once, then API :8000, dashboard :8501, MLflow :5000, weekly monitoring
docker compose run --rm retrain    # retrain with release gates on demand
```

## Monitoring and retraining

`src/monitoring/monitor.py` compares new data with the training data (Evidently) and recomputes performance and fairness (Fairlearn) against the results recorded at go-live. Alerts are amber (review) or red (act). Thresholds and responses are in [docs/monitoring_plan.md](docs/monitoring_plan.md).

`src/models/retrain.py` refits the model on new labelled records and applies four release gates (performance, recall, county fairness, remote communities). A model that passes becomes `challenger` in the MLflow registry. Only a named person can promote it to `champion`:

```bash
python -m src.models.retrain --new-data data/new/inspections.csv
python -m src.models.retrain --promote 2 --approved-by "Name, Operations Manager"
```

## Deployment

- **Dashboard:** Streamlit Community Cloud, entrypoint `app/dashboard.py`, requirements `app/requirements.txt`. Redeploys on push to `main`.
- **API:** Hugging Face Docker Space. `ci.yml` deploys it on push to `main` after tests pass. Needs the secret `HF_TOKEN` and the variable `HF_SPACE` (`<hf-username>/fundifix-api`).

## Governance

Read before using the outputs: [Ethical AI framework](docs/ethical_ai_framework.md) (charter, governance, public trust statement), [Algorithmic Impact Assessment](docs/algorithmic_impact_assessment.md), [Regulatory compliance report](docs/regulatory_compliance.md), [Monitoring plan](docs/monitoring_plan.md) (including incident response and decommissioning), [Reconciliation log](docs/reconciliation_log.md).

## Data and licence

Water Point Data Exchange (2024), WPdx-Plus Kenya, via [OCHA HDX](https://data.humdata.org/dataset/wpdx_ken). Code under Apache 2.0.
