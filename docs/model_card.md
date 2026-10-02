# Model Card: FundiFix water point status classifier

Format follows Mitchell et al. (2019), *Model Cards for Model Reporting*. All figures come from the test set of 1,955 water points unless stated otherwise.

## 1. Model details

| Item | Value |
|---|---|
| Name and version | `fundifix-status-classifier`, version 1, MLflow alias `champion` |
| Type | Random forest, 200 trees, minimum 5 records per leaf, 30% of inputs tried per split, balanced class weights |
| Framework | scikit-learn 1.9.1 pipeline (median fill, scaling, one-hot encoding, forest) |
| File | `models/fundifix_model.joblib`, served by `api/main.py` (`POST /predict`) |
| Owner | Paul (MSc Data Analytics capstone, Nexford University), September 2026 |
| Selection rule | Highest 5-fold cross-validated macro F1 among logistic regression, random forest and XGBoost |

## 2. Intended use

- **Use:** rank rural water points in Kenya by how likely they are to be broken now, so FundiFix can plan inspection visits.
- **Users:** FundiFix operations staff and field supervisors, through the Module 5 dashboard.
- **Decision support only:** a person reviews every visit list. The model never decides that a community will not be served (Module 1 risk ER4).
- **Out of scope:** predicting when a working water point will fail in the future (the data has no failure dates), counties outside the training data, any use to rank, withdraw service from or publicly name communities.

## 3. Training data

- Module 3 clean table, limited to the 4 WPdx sources with mixed labels: 9,773 records from USAID KIWASH (2021), Virtual Kenya (2012 and 2013 to 2014) and Marsabit Water (2012).
- The other 12 sources (12,179 records) label every water point Non-Functional and were excluded. A model trained on them finds only 3.1% of broken water points in the mixed-label test data.
- Stratified 80/20 split: 7,818 training records and 1,955 test records.
- Class shares: Functional 68.8%, Needs Repair 15.0%, Non-Functional 16.3%.
- Inputs (16): age at report, install-year-missing flag, local and assigned population, usage capacity, criticality, pressure, five distances (primary, secondary and tertiary road, city, town), water source, technology, management type, urban or rural.
- Left out: fields only filled for broken points (leakage), data source, report dates, county and GPS (used for fairness checks instead).

## 4. Performance (test set)

| Metric | Baseline (majority) | Selected model | Module 1 target |
|---|---|---|---|
| Balanced accuracy (macro-averaged recall) | 0.333 | 0.567 | 0.80 (G1) |
| Macro F1 | 0.272 | 0.562 | none set |
| ROC-AUC, one class against the rest, macro | 0.500 | 0.774 | none set |
| Recall, Non-Functional (default decision) | 0.000 | 0.376 | 0.70 (G3, reworded) |
| Recall, Non-Functional (visit threshold 0.27) | 0.000 | 0.687 | 0.70 |
| Precision, Non-Functional (visit threshold 0.27) | n/a | 0.272 | none set |

At the 0.27 threshold the model flags 41.1% of water points and finds 68.7% of the broken ones. Visiting the same share at random would find about 41%.

**G1 is not met.** The inputs available in WPdx carry limited signal about a water point's status.

## 5. Fairness

Measured with Fairlearn on the visit decision (Non-Functional probability at or above 0.27). Full tables are in `docs/fairness_report.md`.

| Grouping | Disparate impact ratio | Equalized odds difference | Note |
|---|---|---|---|
| County (10 with 50+ test records) | 0.04 | 0.86 | Three counties have 8 or fewer broken points in the test set |
| Urban / rural | 0.38 | 0.69 | Only 63 urban test records and 1 broken urban point |
| Distance to primary road | 0.64 | 0.20 | Points under 5 km from a road: recall 0.53 vs 0.72 |
| Data source | 0.70 | 0.36 | KIWASH recall 0.75 vs Virtual Kenya 0.39 to 0.55 |

Mitigation with a Fairlearn ThresholdOptimizer (equal recall by county) cut the county equalized odds difference from 0.86 to 0.63, and lowered overall recall from 0.69 to 0.63. It is not adopted by default (see the fairness report).

## 6. Explainability

- Global (SHAP): water source type and pump technology have the largest effect, followed by distance to the primary road, age, and distance to town and city.
- Local: each API response returns the three inputs that moved that water point's Non-Functional probability the most.
- Counterfactual (DiCE): for high-risk points, a change of management type (community to private operator) or technology is the most common change that flips the prediction to Functional. These are associations in the data, not proven causes.

## 7. Limitations and risks

1. **Data source still leaks through.** 97% of Public Tapstand records come from KIWASH, so the top feature partly identifies the survey organisation.
2. **Snapshot data.** Each record has one status at one date. The model estimates current status. It does not forecast failure (Module 1 goal G3 as first written cannot be tested).
3. **Coverage.** Kitui has 33 records in WPdx and Kwale none, and neither is among the mixed-label sources. The model has not seen FundiFix's own counties.
4. **Old records.** Training labels date from 2011 to 2021 (risk ER1).
5. **Small groups.** Fairness results for small counties and urban points rest on very few broken water points.
6. **Stability.** Moving any single input by 20% changes the visit flag for at most 5.3% of water points. 10% random noise on all numeric inputs changes 7.8%.

## 8. Acceptance criteria before use

- Validated on FundiFix's own repair records from Kitui and Kwale, with Non-Functional recall of at least 0.70 at a visit rate FundiFix can staff.
- Recall for every county with 20 or more broken points within 0.15 of the overall recall.
- Human sign-off on every visit list (ER4).

## 9. Monitoring after deployment

| Signal | Alert when |
|---|---|
| Recall on inspected points (share of points found broken that were flagged) | Below 0.60 over a month |
| Share of points flagged | Moves more than 10 points from 41% |
| Input drift (population stability index per input) | Above 0.2 for any top-5 input |
| Share of unknown technology or management values | Above 25% |
| Recall gap between counties | Above 0.15 |

Retrain when FundiFix adds new inspection results, and at least every quarter.
