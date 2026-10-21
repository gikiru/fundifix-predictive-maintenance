# Technical RAID Log

Last reconciled: Final Project (October 2026). See `docs/reconciliation_log.md` for what changed and why.

## Risks

| ID | Risk | Probability | Impact | Mitigation | Owner | Status |
|---|---|---|---|---|---|---|
| R1 | Model staleness from aged WPdx records (training labels 2011 to 2021) | High | High | Drift monitoring (`src/monitoring/monitor.py`); retrain with gates (`src/models/retrain.py`) at least quarterly | Data lead | Mitigated, monitoring live |
| R2 | Equity bias toward accessible water points | Medium | High | Fairness audit by road access: remote points found more often (72% vs 53%). Routing not built | Model lead | Closed for visit list. Reopen if routing is built |
| R3 | GPS, plus-code IDs or enumerator codes could locate communities or people | Medium | High | GPS, notes and activity_id dropped; wpdx_id replaced with neutral IDs in export and dashboard | Data lead | Closed (Final Project) |
| R4 | Risk scores misused to justify service withdrawal | Low | Critical | Human sign-off; use limits in Model Card and Public Trust Statement; incident response plan | Project lead | Open, controlled |
| R5 | Class imbalance (6.7% Needs Repair) degrades minority-class recall | High | Medium | Class weighting; balanced accuracy and macro F1 as main metrics | Model lead | Mitigated |
| R6 | install_year 54% missing limits age-based features | High | Medium | Missing-value flag as feature; do not impute | Data lead | Mitigated |
| R7 | Source label bias: 12 of 16 sources label every point Non-Functional (12,179 records, 55%) | Certain | High | Train on the 4 mixed-label sources; all-sources comparison model kept as evidence | Data lead | Mitigated (found in Module 3) |
| R8 | No coverage of FundiFix counties: 1 Kitui record in training, none in test, no Kwale records | Certain | High | Validate on FundiFix repair records before any routine use | Project lead | Open |
| R9 | County gap in share of broken points found (39% to 86%) | High | Medium | Per-county thresholds tested; trade-off sent to trust funds for decision | Model lead | Open, governance decision |
| R10 | Needs Repair points are not on the visit list, so preventive visits are missed | High | Medium | Add a Needs Repair list after retraining on FundiFix data | Model lead | Open |

## Assumptions

| ID | Assumption | Validated? |
|---|---|---|
| A1 | WPdx Kenya data is operationally analogous to FundiFix's own portfolio | Partially. Same water point types, but not the same counties (see R8) |
| A2 | status_clean is the correct target variable (not status_id) | Yes — cross-checked against admin-level rollup |
| A3 | "Functional, not in use" should be classified as Functional | Yes — decision made in Module 1 |
| A4 | Geographic concentration (8 counties = ~80% of data) is acceptable for a course project | Yes — stated as a limitation, not hidden |

## Issues

| ID | Issue | Severity | Resolution | Status |
|---|---|---|---|---|
| I1 | One record with install_year = 2028 (future date) | Low | Filter during cleaning | Closed (Module 3) |
| I2 | rehab_year, rehabilitator, fecal_coliform fields are 100% missing | Low | Drop entirely | Closed (Module 3) |
| I3 | Streamlit Cloud deployment failed (model file ignored by git, numpy pin on Python 3.14) | Medium | Git exception for model file; version ranges for numpy and pandas | Closed (Module 5) |

## Dependencies

| ID | Dependency | Type | Status |
|---|---|---|---|
| D1 | WPdx Kenya dataset availability on HDX | External | Available |
| D2 | Streamlit Community Cloud free tier | External | Available |
| D3 | Module 1 Vision Document decisions | Internal | Complete |
| D4 | GitHub Free plan and GitHub Actions | External | Available |
| D5 | Hugging Face Spaces free tier (live API) | External | Available |
| D6 | FundiFix repair records for validation | External | Not yet agreed |
