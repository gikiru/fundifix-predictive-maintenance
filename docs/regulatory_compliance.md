# Regulatory Compliance Report

Final Project, October 2026. This is a project assessment, not legal advice. FundiFix should confirm it with counsel before go-live.

## Summary

| Law or framework | Applies? | Why | Status |
|---|---|---|---|
| Kenya Data Protection Act, 2019 | Yes, as the governing law | FundiFix operates in Kenya. The model uses no personal data today, but FundiFix inspection records used for retraining could contain staff or caretaker names | Compliant by design. Actions below |
| Water Act, 2016 (Kenya) | Yes | WASREB oversees water service delivery | Model is decision support only. Service obligations unchanged |
| GDPR (EU) | No | No EU establishment, no EU data subjects, no monitoring of people in the EU (Art. 3) | Principles followed voluntarily |
| CCPA / CPRA (California) | No | FundiFix does not do business in California and processes no California residents' data | Not applicable |
| HIPAA (US) | No | FundiFix is not a covered entity or business associate, and no protected health information is processed. Health facilities appear only as water point users | Not applicable |
| NXU Responsible AI policy | Yes (academic) | Capstone requirement | AI Disclosure Form submitted |

## Kenya Data Protection Act, 2019

| Section | Requirement | How the project meets it |
|---|---|---|
| s.2 Personal data | Information about an identified or identifiable natural person | `notes` (names of people) and `activity_id` (enumerator codes) removed. GPS and plus-code IDs removed from all outputs. Ward names kept: they describe thousands of people, not individuals |
| s.25 Principles | Lawful, minimal, accurate, kept no longer than needed | Only 16 water point attributes used. Raw data fingerprinted. Retention set in `monitoring_plan.md` section 6 |
| s.31 Data protection impact assessment | Needed where processing is likely to result in high risk | Not triggered today (no personal data). The Algorithmic Impact Assessment covers the same questions and must be redone as a DPIA before FundiFix records with personal data are used |
| s.35 Automated individual decision making | People can object to decisions based solely on automated processing that significantly affect them | No decision is solely automated. A supervisor approves every visit list, and no individual is the subject of a score |
| s.43 Notification of breach | Notify the Data Commissioner within 72 hours where there is a real risk of harm | In the incident response plan (I4). Applies if personal data is added later |
| Registration | Data controllers and processors register with the ODPC | FundiFix to confirm its registration covers inspection records |

## GDPR, CCPA and HIPAA: principles adopted anyway

Although none applies, three of their ideas are built in because they reflect good practice:
GDPR Art. 22 (no solely automated decisions with significant effect: human sign-off),
GDPR Art. 35 and CCPA risk assessments (the Algorithmic Impact Assessment), and
HIPAA's minimum necessary standard (only the fields the model needs leave the pipeline).

## Open actions

| ID | Action | Owner | Due |
|---|---|---|---|
| RC1 | Confirm ODPC registration covers inspection and repair records | FundiFix management | Before data collection phase |
| RC2 | Run a DPIA if retraining data includes names or phone numbers | Data lead | Before first retraining on FundiFix data |
| RC3 | Add a data-sharing agreement with the trust funds for repair records | Project lead | Before validation phase |
