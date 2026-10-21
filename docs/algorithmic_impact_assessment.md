# Algorithmic Impact Assessment

System: FundiFix water point risk model, registered as `fundifix-status-classifier`, version 1 (alias `champion`).
Assessed: Final Project, October 2026. Next review: before any use in Kitui or Kwale, and every year after.
Method: questions adapted from the Government of Canada Algorithmic Impact Assessment tool, scored on a 1 (low) to 4 (very high) scale.

## 1. What the system does

| Question | Answer |
|---|---|
| Decision supported | Which water points FundiFix inspects first each week |
| Decision maker | A FundiFix supervisor. The model ranks, the supervisor approves |
| Output | Chance of Functional, Needs Repair or Non-Functional, a visit flag at threshold 0.27, top three reasons |
| Data used | Public WPdx survey attributes of water points. No data about individual people |
| People affected | Households, schools and health facilities served by the water points. Technicians whose routes change |
| Reversibility | High. A missed point is still repaired when the community reports it, as today |
| Scale | Tens to hundreds of water points per week per county |

## 2. Impact scoring

| Area | Score | Reason |
|---|---|---|
| Rights and freedoms | 1 | No individual is assessed. Service rights are protected by the contract and P2 |
| Health and wellbeing | 3 | A missed broken point means more days without safe water, but only relative to a pilot. The current reactive process still runs |
| Economic interests | 2 | Visits are a FundiFix cost. Subscribers pay the same whatever the score |
| Equity | 3 | County gap of 39% to 86% in the share of broken points found |
| Privacy | 1 | No personal data. Location-coded IDs removed (C5) |
| Reversibility and human control | 1 | Human sign-off on every list. Reactive dispatch continues |
| Transparency | 1 | Reasons for every rating, public Model Card and trust statement |
| **Overall level** | **Level 2 (moderate)** | Driven by health and equity. Requires human review, fairness monitoring and an external oversight body, all in place |

## 3. Risks, controls and residual risk

| ID | Risk | Likelihood | Impact | Controls | Residual |
|---|---|---|---|---|---|
| AIA1 | Broken points missed (31% at the default threshold) | High | High | Reactive hotline continues. Threshold slider in dashboard. Recall monitored weekly (act below 60%) | Medium |
| AIA2 | Visits wasted on working points (73% of listed visits) | High | Medium | Ranking still 1.7 times better than random. Hit rate monitored (act below 18 per 100) | Medium |
| AIA3 | Some counties served worse | High | Medium | Bias drift alerts. Per-county thresholds available. Quarterly board review | Medium |
| AIA4 | Model used outside its data (Kitui, Kwale, other sources) | High | High | 81% of inputs drift for non-training sources (monitoring). Validation required before routine use | High until validated |
| AIA5 | Scores used to refuse or delay service | Low | Critical | Contract term, trust statement, incident type I3 | Low |
| AIA6 | Reasons misread as causes | Medium | Medium | Wording "what the model learned", LIME cross-check (first reason agrees 52% of the time) | Medium |
| AIA7 | Unfair retraining data (a source that marks everything broken) | Medium | High | Retraining rejects single-status data. Release gates on fairness | Low |

## 4. Conclusion

The system may be piloted as decision support in one county under the controls above. It must not be used for
routine dispatch in Kitui or Kwale until validated on FundiFix repair records (AIA4). The Ethics Oversight Board
signs this assessment before the pilot starts.

| Role | Name | Signature | Date |
|---|---|---|---|
| Model owner | | | |
| Ethics Oversight Board chair | | | |
