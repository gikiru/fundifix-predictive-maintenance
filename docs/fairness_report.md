# Fairness Metrics Report

**Decision audited:** flag a water point for a visit when its predicted Non-Functional probability is 0.27 or higher.
**Data:** 1,955 test water points (319 broken). Groups with fewer than 50 test records are left out and listed.
**Code:** `src/models/fairness.py`. Group tables: `reports/metrics/fairness_by_*.csv`.

## Metrics used (Module 2, section 6.2)

| Metric | What it measures | Flag when |
|---|---|---|
| Demographic parity difference | Largest gap in the share of points flagged between groups | Reported, not flagged alone (see note) |
| Disparate impact ratio | Lowest flag rate divided by highest flag rate | Below 0.80 (four-fifths rule) |
| Equalized odds difference | Largest gap between groups in recall or false-alarm rate | Above 0.10 |
| Accuracy gap (3 classes) | Largest gap in accuracy between groups (FO1) | Reported |

Note: groups have different shares of broken water points (Nyamira 4.9%, Makueni 26.4%). Equal flag rates would then mean under-serving the groups with more broken points. Module 2 asked for parity "after controlling for actual failure rates", which is what equalized odds measures. It is the main metric here.

## Results by county (FO1)

| County | Test records | Broken | Flag rate | Recall | False-alarm rate | Accuracy (3 classes) |
|---|---|---|---|---|---|---|
| Busia | 324 | 54 | 0.52 | 0.67 | 0.49 | 0.67 |
| Embu | 135 | 23 | 0.33 | 0.39 | 0.31 | 0.84 |
| Kajiado | 95 | 8 | 0.13 | 0.00 | 0.14 | 0.86 |
| Kakamega | 250 | 26 | 0.23 | 0.50 | 0.20 | 0.65 |
| Kiambu | 52 | 5 | 0.17 | 0.20 | 0.17 | 0.90 |
| Kisumu | 315 | 61 | 0.51 | 0.80 | 0.44 | 0.78 |
| Makueni | 296 | 78 | 0.68 | 0.86 | 0.62 | 0.50 |
| Migori | 94 | 15 | 0.47 | 0.80 | 0.41 | 0.50 |
| Nyamira | 164 | 8 | 0.03 | 0.13 | 0.03 | 0.74 |
| Siaya | 152 | 29 | 0.53 | 0.79 | 0.46 | 0.76 |

Left out (under 50 test records): Bungoma, Kericho, Kisii, Machakos, Marsabit, Nairobi, Uasin Gishu, Vihiga.

Disparate impact ratio 0.04, equalized odds difference 0.86, accuracy gap 0.40. **Flagged.**
Recall is lowest in Kajiado, Kiambu and Nyamira, where there are 8 or fewer broken points each. With so few cases one extra catch moves recall by 12 to 20 points, so these rates are uncertain.

## Results by urban or rural (FO2)

| Group | Test records | Broken | Flag rate | Recall | Accuracy (3 classes) |
|---|---|---|---|---|---|
| Rural | 1,892 | 318 | 0.42 | 0.69 | 0.68 |
| Urban | 63 | 1 | 0.16 | 0.00 | 0.94 |

Flagged, but the urban recall is based on a single broken water point and cannot be interpreted.

## Results by distance to primary road (FO2)

| Band | Test records | Broken | Flag rate | Recall | False-alarm rate |
|---|---|---|---|---|---|
| Under 5 km | 517 | 59 | 0.31 | 0.53 | 0.28 |
| 5 to 20 km | 896 | 156 | 0.43 | 0.72 | 0.37 |
| Over 20 km | 542 | 104 | 0.48 | 0.72 | 0.42 |

Disparate impact ratio 0.64, equalized odds difference 0.20. **Flagged.** The gap runs against points near roads, not remote ones. FO2's concern that remote communities would be under-served is not borne out.

## Results by data source (check for residual source bias)

| Source | Test records | Recall | Accuracy (3 classes) |
|---|---|---|---|
| USAID KIWASH 2021 | 1,311 | 0.75 | 0.63 |
| Virtual Kenya 2012 | 500 | 0.55 | 0.82 |
| Virtual Kenya Embu 2013 to 2014 | 135 | 0.39 | 0.84 |

Equalized odds difference 0.36. The model finds broken points best in the KIWASH survey, which also supplies most training records.

## Mitigation

**Method:** Fairlearn ThresholdOptimizer with a true-positive-rate parity constraint by county. It picks a separate threshold for each county. It was fitted on out-of-fold training predictions, so the test set was not used to choose thresholds.

| Measure (10 counties) | Before | After |
|---|---|---|
| Equalized odds difference | 0.86 | 0.63 |
| Disparate impact ratio | 0.04 | 0.36 |
| Overall recall, Non-Functional | 0.69 | 0.63 |
| Precision, Non-Functional | 0.27 | 0.28 |
| Share of points flagged | 0.42 | 0.37 |

Nyamira recall rose from 0.13 to 0.50 and Kakamega from 0.50 to 0.73. Embu fell from 0.39 to 0.26 and Makueni from 0.86 to 0.73.

**Decision:** keep the single threshold as the default and offer per-county thresholds as a dashboard option. The mitigated model finds fewer broken points overall, and the county gaps it corrects rest on 5 to 8 broken points in three counties. Revisit once FundiFix's own inspection data gives enough cases per county.
