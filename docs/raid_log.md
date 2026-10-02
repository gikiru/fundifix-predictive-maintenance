# Technical RAID Log

## Risks

| ID | Risk | Probability | Impact | Mitigation | Owner | Status |
|---|---|---|---|---|---|---|
| R1 | Model staleness from aged WPdx records | High | High | Filter by days_since_report; quarterly retraining | Data lead | Open |
| R2 | Equity bias in routing toward accessible water points | Medium | High | Population-impact weighting; geographic equity audit | Model lead | Open |
| R3 | GPS + population data could identify vulnerable communities | Medium | High | Anonymise identifiers in processed/exported data | Data lead | Open |
| R4 | Risk scores misused to justify service withdrawal | Low | Critical | Human sign-off required; contractual prohibition | Project lead | Open |
| R5 | Class imbalance (6.7% Needs Repair) degrades minority-class recall | High | Medium | SMOTE / class weighting; evaluate macro-F1 not accuracy alone | Model lead | Open |
| R6 | install_year 54% missing limits age-based features | High | Medium | Missing-value flag as feature; do not impute | Data lead | Open |

## Assumptions

| ID | Assumption | Validated? |
|---|---|---|
| A1 | WPdx Kenya data is operationally analogous to FundiFix's own portfolio | Partially — same water point types and counties |
| A2 | status_clean is the correct target variable (not status_id) | Yes — cross-checked against admin-level rollup |
| A3 | "Functional, not in use" should be classified as Functional | Yes — decision made in Module 1 |
| A4 | Geographic concentration (8 counties = ~80% of data) is acceptable for a course project | Yes — stated as a limitation, not hidden |

## Issues

| ID | Issue | Severity | Resolution | Status |
|---|---|---|---|---|
| I1 | One record with install_year = 2028 (future date) | Low | Filter during cleaning | Open |
| I2 | rehab_year, rehabilitator, fecal_coliform fields are 100% missing | Low | Drop entirely | Open |

## Dependencies

| ID | Dependency | Type | Status |
|---|---|---|---|
| D1 | WPdx Kenya dataset availability on HDX | External | Available |
| D2 | Streamlit Community Cloud free tier | External | Available |
| D3 | Module 1 Vision Document decisions | Internal | Complete |
| D4 | GitHub Free plan | External | Available |
