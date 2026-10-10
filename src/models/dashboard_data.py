"""
Build the data files the Streamlit dashboard reads (app/data/).

The dashboard is public, so these files follow the Module 3 privacy rules:
no GPS points, no WPdx IDs (they are plus codes that encode location)
and no water point names. Place names (county, sub-county, ward) are kept.
Only the 1,955 test-set water points are included, so every number the dashboard shows
comes from records the model did not see during training.

Run from repo root after training:  python -m src.models.dashboard_data
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap

from src.models.config import CLASSES, FEATURES, METRICS_DIR, MODEL_DIR, MODEL_PATH
from src.models.data import load_modelling_data, split
from src.models.explain import LABELS, original_feature

OUT = MODEL_DIR.parent / "app" / "data"
NF = CLASSES.index("Non-Functional")
SOURCE_SHORT = {
    "USAID_KIWASH_Kenya_2021": "USAID KIWASH 2021",
    "Virtual Kenya_Busia, Kajiado, Kiambu, Kisumu_2012": "Virtual Kenya 2012",
    "Virtual Kenya_Embu Kenya_2013-2014": "Virtual Kenya Embu 2013-14",
    "Marsabit Water_Kenya_2012": "Marsabit Water 2012",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL_PATH)
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
    _, test = split(load_modelling_data("mixed"))

    proba = model.predict_proba(test[FEATURES])
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    X_t = pd.DataFrame(prep.transform(test[FEATURES]), columns=prep.get_feature_names_out())
    sv = shap.TreeExplainer(clf)(X_t).values[:, :, NF]
    groups = [original_feature(c) for c in X_t.columns]
    per_feature = pd.DataFrame(sv, columns=X_t.columns).T.groupby(groups).sum().T

    reasons = []
    for i in range(len(per_feature)):
        row = per_feature.iloc[i]
        top = row.reindex(row.abs().sort_values(ascending=False).index)[:3]
        reasons.append(" | ".join(f"{LABELS[f]} ({'raises' if v > 0 else 'lowers'} risk)"
                                  for f, v in top.items()))

    out = pd.DataFrame({
        # wpdx_id is an Open Location Code (plus code) that pinpoints the water point to a few
        # metres, so it is replaced with a neutral ID in this public file.
        "point_id": [f"WP-{i + 1:04d}" for i in range(len(test))],
        "county": test["clean_adm1"],
        "sub_county": test["clean_adm2"],
        "ward": test["clean_adm3"],
        "data_source": test["dataset_title"].map(SOURCE_SHORT),
        "road_access": test["road_access"],
        "actual_status": test["target"],
        "predicted_status": [CLASSES[k] for k in proba.argmax(1)],
        "p_functional": proba[:, 0].round(4),
        "p_needs_repair": proba[:, 1].round(4),
        "p_non_functional": proba[:, NF].round(4),
        "top_reasons": reasons,
    })
    for f in FEATURES:
        out[f] = test[f].to_numpy()
    for f in per_feature.columns:
        out[f"shap_{f}"] = per_feature[f].round(4).to_numpy()
    out.to_csv(OUT / "water_points.csv", index=False)

    s = json.loads((METRICS_DIR / "training_summary.json").read_text())
    summary = {
        "model_version": meta["version"], "family": meta["family"], "nf_threshold": meta["nf_threshold"],
        "features": FEATURES, "labels": LABELS, "classes": CLASSES,
        "test_rows": len(out), "train_rows": s["rows"]["train"],
        "test_metrics": {k: s["results"][k]["test"] for k in
                         ["baseline_majority", "logistic_regression", "random_forest", "xgboost"]},
        "operating_point": s["results"][s["selected_model"]]["operating_point"],
        "all_sources_comparison": s["results"]["comparison_all_sources"],
        "global_drivers": pd.read_csv(METRICS_DIR / "shap_global_importance.csv", index_col=0)
                            .rename(index=LABELS)["all_classes"].round(4).to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"Wrote {len(out):,} water points and summary to {OUT}")


if __name__ == "__main__":
    main()
