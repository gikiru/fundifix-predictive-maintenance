# Ethical AI Framework: FundiFix Water Point Risk Model

Version 1.0, Final Project (October 2026). Owner: project lead. Review: every quarter, and after any red alert or incident.
This document brings together the ethics work from Modules 1 to 5 and the controls added in the Final Project.

## 1. Ethical AI Charter (from Module 1, reconciled)

| ID | Principle | What it means in practice | Where it is enforced |
|---|---|---|---|
| P1 | Decision support, not decision making | The model ranks water points. A FundiFix supervisor approves every visit list. | Dashboard and API notices. Use limits in the Model Card. |
| P2 | No one loses service because of a score | A low score never removes or delays a contracted repair. The three-day guarantee is a human commitment. | Public Trust Statement (section 5). Incident type I3 (`monitoring_plan.md`). |
| P3 | Explainable to non-specialists | Every rating comes with its top three reasons. Reasons are checked with a second method. | SHAP in API and dashboard. LIME cross-check (`src/models/lime_check.py`). |
| P4 | Fair across places | Broken points are found at similar rates across counties and road access. Remote communities are never worse off. | Fairlearn audit (Module 4), monitoring bias drift, release gate "remote". |
| P5 | Privacy by design | No personal data in outputs. No field that locates a water point to a few metres. | Pipeline anonymisation, neutral IDs, hashed references in the explanation store. |
| P6 | Accountable and traceable | Every model version, prediction and promotion can be traced to data, code and a named person. | MLflow registry, explanation store, model change log, audit log. |
| P7 | Honest about limits | Unmet goals and failed mitigation are reported, not reworded. | Model Card, reconciliation log, dashboard Limits tab. |

Change from Module 1: the charter said village and ward names would be anonymised. Ward and sub-county names are public place names, not personal data, and are kept (reconciliation item C4). The Final Project added P6 and P7 as explicit principles.

## 2. How the ethics work connects across modules

| Ethical risk (Module 1) | Found or tested in | Control now in place | Status |
|---|---|---|---|
| ER1 Model staleness | M3 audit: labels 2011 to 2021 | Weekly drift monitoring, retraining with gates | Controlled |
| ER2 Equity bias | M3 bias check, M4 Fairlearn audit | Road-access check in monitoring and release gates | Remote points found more often (72% vs 53%) |
| ER3 Community privacy | M3 anonymisation, M5 found plus-code IDs | Neutral IDs in export, dashboard and reports | Closed |
| ER4 Misuse of scores | M4 Model Card use limits | Sign-off, trust statement, incident response | Open, controlled |
| New: source label bias (R7) | M3 bias check | Training limited to 4 mixed-label sources. Retraining rejects single-status data | Controlled |
| New: no FundiFix-county data (R8) | M3 and M4 | Validation on FundiFix records required before routine use | Open |
| New: county gap (R9) | M4 Fairlearn | Per-county thresholds tested. Decision referred to trust funds | Open, governance decision |

## 3. Governance structure

| Body or role | Who | Responsibilities | Meets |
|---|---|---|---|
| Ethics Oversight Board | Kwale and Kitui Water Services Maintenance Trust Funds, FundiFix management | Approves go-live, fairness trade-offs (for example per-county thresholds), decommissioning. Reviews the quarterly fairness report. | Quarterly, and within 5 working days of a red ethics incident |
| Model owner | FundiFix data lead | Runs monitoring, retraining and incident response. Keeps the Model Card current. | Weekly review of monitoring |
| Approver for promotion | FundiFix operations manager | Promotes a gate-passing challenger to champion with `--approved-by` | As needed |
| Field supervisors | FundiFix dispatch coordinators | Review and sign off each weekly visit list. Report wrong or unfair lists. | Weekly |
| Communities and institutions | Subscribers, through the FundiFix hotline | Raise concerns about service. Can ask why a point was or was not visited. | Any time |
| Regulator | WASREB (Water Act, 2016) | External oversight of service delivery | Per WASREB reporting cycle |
| Data protection | Office of the Data Protection Commissioner (ODPC) | Receives breach notifications if personal data is ever involved | As required |

Decision rights. The model owner can pause the visit list (fall back to reactive dispatch) without approval. Only the Ethics Oversight Board can change the fairness rule, the visit threshold policy, or retire the model.

## 4. Lifecycle controls

1. **Data:** raw files fingerprinted, 12 automated quality checks, bias check before training, anonymised export.
2. **Model:** cross-validated selection, fixed hold-out test, fairness audit, sensitivity test, MLflow registry with aliases.
3. **Release:** four automated gates (performance, recall, county fairness, remote communities) and named human approval.
4. **Operation:** every prediction logged with reasons, weekly drift and bias-drift monitoring, weekly uptime check.
5. **Retirement:** decommissioning plan in `monitoring_plan.md`, section 6.

## 5. Public Trust Statement

> FundiFix uses a computer model to help plan inspection visits to rural water points. The model looks at
> public survey information about each water point, such as its water source, pump type and distance to roads,
> and estimates how likely it is to be broken now. It does not use any information about individual people.
>
> The model helps us decide where to look first. It does not decide who gets a repair. Every contracted water
> point keeps the three-day repair guarantee, whatever its score. A FundiFix supervisor checks every visit list
> before teams go out, and the hotline stays open for every community.
>
> We check every quarter that the model finds broken water points as well in remote areas as near main roads,
> and across counties. We publish where it falls short. Our results so far: the model finds remote broken
> points more often than nearby ones, but some counties are served better than others. The county trust funds
> review these results and decide how to respond.
>
> The model has not yet been tested on records from Kitui or Kwale. We will not use it for routine planning there
> until it has been.
>
> Questions or concerns: call the FundiFix hotline or write to the county trust fund. You can ask why a water point
> was or was not on a visit list, and we will explain the main reasons.

## 6. Related documents

`model_card.md`, `fairness_report.md`, `algorithmic_impact_assessment.md`, `regulatory_compliance.md`,
`monitoring_plan.md`, `raid_log.md`, `reconciliation_log.md`.
