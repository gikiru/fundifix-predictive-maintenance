"""
Module 4 sensitivity analysis: how much do predictions change when inputs change a little?
Each numeric input is moved by -20%, -10%, +10% and +20% on the test set, one at a time.
Age is moved by -2, -1, +1 and +2 years instead. Then all numeric inputs get 10% random noise together.
Run from repo root after training:  python -m src.models.sensitivity
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models import plots
from src.models.config import CLASSES, FEATURES, FIG_DIR, METRICS_DIR, MODEL_PATH, NUMERIC_FEATURES, RANDOM_STATE
from src.models.data import load_modelling_data, split
from src.models.explain import LABELS

NF = CLASSES.index("Non-Functional")


def compare(model, base_X, new_X, threshold):
    p0, p1 = model.predict_proba(base_X), model.predict_proba(new_X)
    return {"class_changed": float((p0.argmax(1) != p1.argmax(1)).mean()),
            "flag_changed": float(((p0[:, NF] >= threshold) != (p1[:, NF] >= threshold)).mean()),
            "mean_abs_change_p_nf": float(np.abs(p1[:, NF] - p0[:, NF]).mean())}


def main():
    model = joblib.load(MODEL_PATH)
    s = json.loads((METRICS_DIR / "training_summary.json").read_text())
    threshold = s["results"][s["selected_model"]]["operating_point"]["nf_threshold"]
    _, test = split(load_modelling_data("mixed"))
    X = test[FEATURES].copy()

    rows = []
    for col in NUMERIC_FEATURES:
        if col == "install_year_missing":
            continue
        steps = [-2, -1, 1, 2] if col == "age_at_report" else [-0.2, -0.1, 0.1, 0.2]
        for step in steps:
            Xn = X.copy()
            Xn[col] = Xn[col] + step if col == "age_at_report" else Xn[col] * (1 + step)
            rows.append({"feature": col, "change": step, **compare(model, X, Xn, threshold)})
    Xn = X.copy()
    Xn["install_year_missing"] = 1
    Xn["age_at_report"] = np.nan
    rows.append({"feature": "install_year_missing", "change": "set to unknown", **compare(model, X, Xn, threshold)})
    rng = np.random.default_rng(RANDOM_STATE)
    Xn = X.copy()
    for col in NUMERIC_FEATURES:
        if col != "install_year_missing":
            Xn[col] = Xn[col] * (1 + rng.normal(0, 0.10, len(Xn)))
    rows.append({"feature": "all numeric inputs", "change": "10% random noise", **compare(model, X, Xn, threshold)})

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "sensitivity_analysis.csv", index=False)

    worst = (df[df["change"].isin([-0.2, 0.2, -2, 2])].groupby("feature")["flag_changed"].max()
             .sort_values(ascending=True))
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.barh([LABELS.get(f, f) for f in worst.index], worst.values, color=plots.TEAL, height=0.6)
    for i, v in enumerate(worst.values):
        ax.text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=8)
    ax.set_xlabel("Share of test water points whose visit flag changes")
    ax.set_title("Sensitivity: each input moved by 20% (age by 2 years)", fontsize=10, loc="left")
    ax.set_xlim(0, max(0.05, worst.max() * 1.25))
    fig.savefig(FIG_DIR / "sensitivity_flag_changes.png")
    plt.close(fig)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
