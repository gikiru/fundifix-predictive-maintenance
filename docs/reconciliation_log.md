# Cross-Module Reconciliation Log

Audit of Modules 1 to 5 before the final integrated report. Each row is a place where an earlier
document said one thing and the built system does another. The "Resolution" column says which
version is now true and where the change was made.

| ID | Where it was said | What the project actually did | Resolution |
|---|---|---|---|
| C1 | M1 G1: 80% macro-averaged accuracy | Balanced accuracy 0.567 (random forest), 0.589 (XGBoost) | Reported as not met. Not reworded. |
| C2 | M1 G3: flag 70% of points "before failure" by back-testing | Data has one status per point and no repair history, so back-testing failures is impossible | G3 restated as "flag at least 70% of currently broken points". Result 68.7%. Both wordings shown in the report. |
| C3 | M1 ER1: data spans 2011 to 2024 | Usable (mixed-label) training labels span 2011 to 2021 | Model Card and report use 2011 to 2021. |
| C4 | M1 and M2 privacy plan: anonymise village and ward names | Ward and sub-county names are public place names, not personal data under the Kenya Data Protection Act | Kept on purpose. Reason documented in `src/data/anonymize.py`. |
| C5 | M3 export: GPS and notes removed | `wpdx_id` is a plus code that locates a point to a few metres, and `activity_id` holds enumerator codes | Fixed in Final Project: export now replaces `wpdx_id` with neutral IDs and drops `activity_id` (`src/data/anonymize.py`, tested). Plus codes also removed from `reports/metrics/explainability_summary.json`, the evaluation notebook output and the API example. |
| C6 | M1 risk register ER1 to ER4 | M3 bias check found a larger risk: 12 of 16 sources label every point broken (55% of records) | Added to RAID log as R7. Model trained on the 4 mixed-label sources only. |
| C7 | M1 and M2: Kitui and Kwale are the target counties | Training data has 1 Kitui record and 0 Kwale records. Test data has none | Added to RAID log as R8. Model Card, dashboard and report now say "one Kitui record, none in test". Earlier text said "no records". |
| C8 | M1 solution: prescriptive routing layer and CSV/PDF export | Not built in Modules 3 to 5 | CSV export of the visit list added to the dashboard in Final Project. Routing and PDF remain out of scope and are listed as limitations. |
| C9 | M2: SMOTE or class weighting | Class weighting (`class_weight="balanced"`) only | Report says class weighting. |
| C10 | M1 stack: DVC for data versioning | DVC was not set up. Raw files are fixed HDX downloads | Raw files are now fingerprinted (SHA-256) in `data/raw/MANIFEST.json` and checked at the start of every pipeline run. DVC listed as not implemented. |
| C11 | Course matrix: Airflow DAG for the pipeline | Prefect flow, chosen in M1 as the free self-hosted option | Kept Prefect. Same role (scheduled, logged, retried tasks). |
| C12 | M1: Git Flow branches and pull requests | Work committed to `main` | CI now runs on every push and pull request. Branching stated as simplified. |
| C13 | M1: semantic model versions (1.0.0) | MLflow registry integer versions with aliases (`champion`, `challenger`) | MLflow versions are the record. Report uses them. |
| C14 | M4 Model Card: explanations "available" | Explanations shown on screen but not stored | Explanation store added: every API prediction and its top reasons are appended to `data/explanation_store.duckdb`. |
| C15 | M4: "Live API" required | API ran only locally | Deployed as a Render free web service. CI triggers the deploy on push to `main` after tests pass. (Hugging Face Docker Spaces were tried first but now require a paid plan.) |
| C16 | M5 dashboard: 7 tabs, no monitoring | Brief needs monitoring of performance, drift and fairness | Monitoring tab added, fed by `src/monitoring/monitor.py` (Evidently and Fairlearn). |
| C17 | M1 FO1: equal accuracy across regions | County gap 39% to 86% in share found. Mitigation narrowed it but cut overall recall | Reported as not met, with the trade-off. Left as a governance decision for the trust funds. |
| C18 | M1 NFR: works on 3G | Dashboard uses precomputed data and caching, but was not tested on a throttled link | Stated as untested. |
| C19 | M4: explanations with SHAP and DiCE | Brief also lists LIME | LIME check added (`src/models/lime_check.py`) to compare with SHAP on the same points. |
| C20 | M4 Model Card monitoring section: population stability index, county gap above 0.15 | Implemented monitoring uses Evidently drift tests and bias drift against go-live | Model Card section 9 rewritten to match `src/monitoring/monitor.py`. |
| C21 | M4: SHAP reasons presented as the explanation | LIME agrees with SHAP on the first reason for 52% of points | Model Card and dashboard advise treating the first reason as the most reliable. |
| C22 | M4 fairness report: county equalized odds difference 0.86 (before mitigation) | Monitoring computes 0.47 for county | Different group rules, both stated: Module 4 compared counties with 50+ test records. Monitoring also requires 10+ broken points, which removes three counties whose rates rest on 5 to 8 broken points. Report gives both. |
