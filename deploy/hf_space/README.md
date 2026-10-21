---
title: FundiFix Water Point Status API
emoji: 💧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Ranks rural water points by chance of being broken now
---

# FundiFix water point status API

Decision-support API from the FundiFix Predictive Maintenance capstone (Nexford University, MSc Data Analytics).
For each water point it returns the chance of being Functional, Needs Repair or Non-Functional,
a visit flag at the approved threshold, and the top three reasons (SHAP).

- `GET /health` and `GET /model-info`
- `POST /predict` (interactive docs at `/docs`)
- `GET /explanations/{prediction_id}`: the stored inputs, outputs and reasons for a past prediction
- `GET /audit/summary`

Trained on public WPdx-Plus Kenya survey data. Not validated in Kitui or Kwale. A FundiFix supervisor
reviews every visit list. Scores must not be used to reduce or withdraw service from any community.
On this free Space the explanation store resets when the Space restarts.

Code, Model Card and governance documents: https://github.com/gikiru/fundifix-predictive-maintenance
