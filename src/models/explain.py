"""
Module 4 explainability: SHAP global and local explanations, and DiCE counterfactuals.
Run from repo root after training:  python -m src.models.explain
"""
import json
import random
import warnings

import dice_ml
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models import plots
from src.models.config import (CATEGORICAL_FEATURES, CLASSES, FEATURES, FIG_DIR, METRICS_DIR,
                               MODEL_PATH, NUMERIC_FEATURES, RANDOM_STATE)
from src.models.data import load_modelling_data, split

warnings.filterwarnings("ignore")
NF = CLASSES.index("Non-Functional")

LABELS = {
    "age_at_report": "Age at report (years)",
    "install_year_missing": "Install year not recorded",
    "local_population": "Population within 1 km",
    "assigned_population": "Population assigned to point",
    "usage_cap": "Usage capacity (people)",
    "criticality": "Criticality score",
    "pressure": "Demand pressure",
    "distance_to_primary": "Distance to primary road",
    "distance_to_secondary": "Distance to secondary road",
    "distance_to_tertiary": "Distance to tertiary road",
    "distance_to_city": "Distance to city",
    "distance_to_town": "Distance to town",
    "water_source_clean": "Water source type",
    "water_tech_clean": "Pump / technology type",
    "management_clean": "Management type",
    "is_urban": "Urban or rural",
}


def original_feature(name: str) -> str:
    """Map a one-hot column (e.g. water_tech_clean_Hand Pump) back to its source feature."""
    for c in CATEGORICAL_FEATURES:
        if name.startswith(c + "_"):
            return c
    return name


def readable(name: str) -> str:
    base = original_feature(name)
    if base == name:
        return LABELS.get(name, name)
    return f"{LABELS[base]}: {name[len(base) + 1:]}"


def shap_values(model, X):
    """TreeExplainer on the fitted forest. Returns (values[n, features, classes], names, X_t)."""
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    X_t = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out())
    explainer = shap.TreeExplainer(clf)
    sv = explainer(X_t)
    return sv, X_t


def global_importance(sv, names) -> pd.DataFrame:
    """Mean absolute SHAP per original feature, per class.
    One-hot columns of the same feature are summed per row first, then made absolute."""
    groups = pd.Series([original_feature(n) for n in names])
    out = {}
    for k, cls in enumerate(CLASSES):
        per_row = pd.DataFrame(sv.values[:, :, k], columns=names).T.groupby(groups.values).sum().T
        out[cls] = per_row.abs().mean()
    g = pd.DataFrame(out)
    g["all_classes"] = g.sum(axis=1)
    return g.sort_values("all_classes", ascending=False)


def plot_global(g: pd.DataFrame, path):
    top = g.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    left = np.zeros(len(top))
    for cls, col in zip(CLASSES, [plots.NAVY, plots.TEAL, plots.TERRA]):
        ax.barh([LABELS.get(i, i) for i in top.index], top[cls], left=left, color=col,
                label=cls, height=0.6, edgecolor="white", linewidth=1)
        left += top[cls].to_numpy()
    ax.set_xlabel("Mean |SHAP value| (average effect on predicted probability)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Global feature importance by class (test sample)", fontsize=10, loc="left")
    fig.savefig(path)
    plt.close(fig)


def plot_beeswarm_nf(sv, X_t, path):
    exp = shap.Explanation(values=sv.values[:, :, NF], base_values=sv.base_values[:, NF],
                           data=X_t.to_numpy(), feature_names=[readable(c) for c in X_t.columns])
    plt.figure()
    shap.plots.beeswarm(exp, max_display=12, show=False, plot_size=(7, 4.6))
    plt.title("How each feature pushes the Non-Functional probability up or down", fontsize=10, loc="left")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")


def display_values(X_t, raw):
    """Raw (unscaled) values for display. Distances shown in km."""
    d = X_t.copy()
    for c in NUMERIC_FEATURES:
        v = raw[c].to_numpy(dtype=float)
        d[c] = np.round(v / 1000, 1) if c.startswith("distance_") else v
    return d


def plot_waterfall(sv, X_disp, i, title, path):
    names = [readable(c) + (" (km)" if c.startswith("distance_") else "") for c in X_disp.columns]
    exp = shap.Explanation(values=sv.values[i, :, NF], base_values=sv.base_values[i, NF],
                           data=X_disp.iloc[i].to_numpy(), feature_names=names)
    plt.figure()
    shap.plots.waterfall(exp, max_display=9, show=False)
    plt.title(title, fontsize=10, loc="left")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")


def counterfactuals(model, train, query, n=3):
    """DiCE: smallest changes to actionable features that turn a Non-Functional prediction Functional."""
    fill = {c: train[c].median() for c in NUMERIC_FEATURES}
    tr = train[FEATURES].fillna(fill).assign(y=train["y"].to_numpy())
    q = query[FEATURES].fillna(fill)
    data = dice_ml.Data(dataframe=tr, continuous_features=NUMERIC_FEATURES, outcome_name="y")
    m = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
    dice = dice_ml.Dice(data, m, method="random")
    # Only inputs FundiFix or a water committee can change, limited to real options.
    # Usage capacity and pressure are left fixed: WPdx derives them from the technology and population.
    vary = ["management_clean", "water_tech_clean"]
    allowed = {
        "management_clean": ["Community Management", "Private Operator/Delegated Management"],
        "water_tech_clean": sorted(t for t in train["water_tech_clean"].unique() if t != "Unknown"),
    }
    random.seed(RANDOM_STATE)         # DiCE also draws from the global random generators
    np.random.seed(RANDOM_STATE)
    res = dice.generate_counterfactuals(q, total_CFs=n, desired_class=CLASSES.index("Functional"),
                                        features_to_vary=vary, permitted_range=allowed,
                                        random_seed=RANDOM_STATE)
    out = []
    for i, cf in enumerate(res.cf_examples_list):
        orig = q.iloc[i]
        for _, row in cf.final_cfs_df.iterrows():
            changes = {LABELS[c]: {"from": orig[c], "to": row[c]} for c in vary if str(orig[c]) != str(row[c])}
            out.append({"water_point": int(i), "changes": changes})
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL_PATH)
    train, test = split(load_modelling_data("mixed"))
    sample = test.sample(n=min(600, len(test)), random_state=RANDOM_STATE).reset_index(drop=True)
    sv, X_t = shap_values(model, sample[FEATURES])

    g = global_importance(sv, list(X_t.columns))
    g.to_csv(METRICS_DIR / "shap_global_importance.csv")
    plot_global(g, FIG_DIR / "shap_global_importance.png")
    plot_beeswarm_nf(sv, X_t, FIG_DIR / "shap_beeswarm_non_functional.png")

    # Local explanations: one Non-Functional point the model caught, one it missed
    p_nf = model.predict_proba(sample[FEATURES])[:, NF]
    nf = sample["y"].to_numpy() == NF
    caught = int(np.argmax(np.where(nf, p_nf, -1)))
    missed = int(np.argmin(np.where(nf, p_nf, 2)))
    X_disp = display_values(X_t, sample)
    plot_waterfall(sv, X_disp, caught, f"Local explanation: broken point the model caught (P = {p_nf[caught]:.2f})",
                   FIG_DIR / "shap_local_caught.png")
    plot_waterfall(sv, X_disp, missed, f"Local explanation: broken point the model missed (P = {p_nf[missed]:.2f})",
                   FIG_DIR / "shap_local_missed.png")

    # Counterfactuals for the three points with the highest Non-Functional probability
    top = sample.iloc[np.argsort(-p_nf)[:3]]
    cfs = counterfactuals(model, train, top)

    summary = {"sample_size": len(sample),
               "top_features": g["all_classes"].head(8).round(4).to_dict(),
               "top_features_non_functional": g["Non-Functional"].sort_values(ascending=False).head(8).round(4).to_dict(),
               "local": {"caught": {"p_nf": float(p_nf[caught]), "wpdx_id": sample.loc[caught, "wpdx_id"]},
                         "missed": {"p_nf": float(p_nf[missed]), "wpdx_id": sample.loc[missed, "wpdx_id"]}},
               "counterfactual_queries": top[["wpdx_id"] + FEATURES].astype(str).to_dict(orient="records"),
               "counterfactual_query_p_nf": [float(x) for x in np.sort(-p_nf)[:3] * -1],
               "counterfactuals": cfs}
    (METRICS_DIR / "explainability_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ["top_features", "top_features_non_functional", "local"]}, indent=1, default=str))
    print(f"{len(cfs)} counterfactuals written")


if __name__ == "__main__":
    main()
