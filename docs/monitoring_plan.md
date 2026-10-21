# Implementation and Monitoring Plan

Final Project, October 2026. Code: `src/monitoring/monitor.py`, `src/flows/mlops_flow.py`, `src/models/retrain.py`.
Results: dashboard Monitoring tab (`app/data/monitoring/`).

## 1. Deployment plan

| Phase | When | What | Exit condition |
|---|---|---|---|
| 0. Live demo | Now | API on Hugging Face, dashboard on Streamlit Cloud, CI/CD on GitHub Actions | Tests pass, uptime check green |
| 1. Data | Months 1 to 3 | FundiFix records every inspection result in a dated, consistent format | Records for every visit in Kitui and Kwale |
| 2. Validate | Months 3 to 4 | Score FundiFix records. Compare with what teams found | At least 65% of broken points found (gate) |
| 3. Pilot | Months 4 to 6 | One county uses the visit list beside current practice | Shorter downtime, fewer visits per broken point found |
| 4. Retrain | Month 6 | Retrain on FundiFix data through the release gates | Challenger passes all gates and is approved |
| 5. Operate | Ongoing | Weekly monitoring, quarterly board review | No unresolved red alerts |

Production note: the free tiers sleep when idle and the Hugging Face Space loses its explanation store on restart.
For routine use, the API should move to an always-on host with the store on persistent storage.

## 2. Performance KPIs

| KPI | Go-live value (test set) | Review (amber) | Act (red) |
|---|---|---|---|
| Share of broken points found (recall) | 69% | below 65% | below 60% |
| Broken points per 100 visits | 27 | below 22 | below 18 |
| Balanced accuracy | 0.567 | drop of 0.03 | drop of 0.05 |
| Share of points flagged | 41% | change of 10 points | change of 20 points |
| Inputs drifted from training data | 0% | 25% | 50% |
| API uptime | n/a | one failed weekly check | two in a row |
| Operational (pilot): days from fault to repair | FundiFix baseline | no improvement after 3 months | worse than baseline |

## 3. Continuous fairness monitoring

Every weekly run recomputes, with Fairlearn, for county, road access and urban/rural:
share of broken points found per group, equalized odds difference, demographic parity difference and disparate impact ratio.
Only groups with at least 50 records and 10 broken points are compared, so small groups do not trigger false alarms.

| Check | Amber | Red |
|---|---|---|
| Gap in share found between groups (bias drift) | rises 5 points above go-live | rises 10 points |
| Remote points (over 20 km) found less often than near points (under 5 km) | n/a | any week |

Quarterly, the model owner sends the fairness results and the county gap to the Ethics Oversight Board.

## 4. Model drift and bias drift detection

| Type | Method | Reference |
|---|---|---|
| Data drift | Evidently `DataDriftPreset`: K-S or chi-square for small samples, Wasserstein or Jensen-Shannon distance for large ones, per input | Training split (7,818 records) |
| Prediction drift | Share flagged and mean chance of being broken | Go-live baseline |
| Performance drift | Recall, hits per 100 visits, balanced accuracy, on any batch with confirmed statuses | Go-live baseline |
| Bias drift | Change in group gaps (section 3) | Go-live baseline |
| Explanation check | LIME versus SHAP agreement on the first reason (52% at go-live) | Re-run at each retraining |

Batches monitored each week: the hold-out test set (control), WPdx records from sources not used in training,
a simulated stress test, and live API requests from the explanation store (once there are 30).
First results: control green, other sources red (81% of inputs drift), stress test amber.

Retraining runs when new labelled records arrive in `data/new/` or the control batch turns red. A retrained model
must pass four gates and a named approval before it replaces the champion. The previous champion is kept for rollback.

## 5. Incident response plan for ethical failures

| Type | Example | Severity |
|---|---|---|
| I1 Fairness failure | Remote points found less often than near points, or bias drift red | High |
| I2 Performance failure | Recall below 60% on confirmed results | High |
| I3 Misuse | A score used to delay or refuse a contracted repair | Critical |
| I4 Privacy | A location-coded ID or a person's name appears in an output | Critical |
| I5 Service | API or dashboard down two weeks running | Medium |

Steps:
1. **Detect** (any time): monitoring alert, uptime issue, supervisor report or community complaint through the hotline.
2. **Contain** (within 24 hours for High and Critical): the model owner pauses the visit list. Teams return to reactive dispatch, which keeps the repair guarantee intact. For I4, remove the output and rotate any exposed files.
3. **Notify** (within 72 hours): Ethics Oversight Board for all High and Critical incidents. ODPC within 72 hours if personal data was exposed (Data Protection Act, s.43). Affected communities if a repair was delayed.
4. **Investigate** (within 10 working days): use the explanation store to reconstruct each affected prediction (inputs, version, threshold, reasons). Find the cause: data, model, threshold or human use.
5. **Fix and review**: roll back to `previous_champion`, retrain through the gates, or change the procedure. The board approves restart.
6. **Learn**: record the incident and the fix in the RAID log and Model Card.

## 6. Decommissioning plan

Triggers: the board decides the model no longer helps (pilot shows no gain), a critical incident cannot be fixed,
the data source ends, or a better system replaces it.

1. The board approves retirement and a last date of use. Field teams are told two weeks ahead.
2. Dispatch returns to the reactive process, or moves to the replacement system, with no gap in the repair guarantee.
3. The API and dashboard are taken down, and the Hugging Face Space and Streamlit app are deleted. The uptime workflow is disabled.
4. The model version is archived in MLflow with the tag `retired` and the reason. The model file is kept for 2 years for audit, then deleted.
5. The explanation store is kept read-only for 2 years so past decisions can still be explained, then deleted. Any personal data is deleted earlier if no longer needed.
6. A closing note is added to the Model Card and the public trust statement, saying what the model did, why it was retired and what replaced it.
