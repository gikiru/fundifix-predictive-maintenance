"""
LIME cross-check of the SHAP explanations (Final Project).

SHAP is the explanation shown to users. LIME uses a different method (a simple local model
fitted to perturbed copies of one water point). If the two agree on the main reasons,
supervisors can trust the reasons more. If they disagree, the reason is reported as uncertain.

Run from repo root after training:  python -m src.models.lime_check
Outputs: reports/figures/lime_local_caught.png, reports/metrics/lime_shap_agreement.json
"""
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from src.models import plots
from src.models.config import (CATEGORICAL_FEATURES, CLASSES, FEATURES, FIG_DIR, METRICS_DIR, MODEL_PATH,
                               NUMERIC_FEATURES, RANDOM_STATE)
from src.models.data import load_modelling_data, split
from src.models.explain import LABELS, original_feature, shap_values

warnings.filterwarnings("ignore")
NF = CLASSES.index("Non-Functional")
N_POINTS = 100


def encoder(train):
    """LIME needs numbers. Categories become integer codes, missing numbers become training medians."""
    cats = {c: sorted(train[c].astype(str).unique()) for c in CATEGORICAL_FEATURES}
    med = train[NUMERIC_FEATURES].median()

    def encode(df):
        out = df[FEATURES].copy()
        out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].fillna(med)
        for c in CATEGORICAL_FEATURES:
            out[c] = out[c].astype(str).map({v: i for i, v in enumerate(cats[c])}).fillna(0)
        return out.to_numpy(dtype=float)

    def decode(arr):
        df = pd.DataFrame(arr, columns=FEATURES)
        for c in CATEGORICAL_FEATURES:
            df[c] = [cats[c][int(round(min(max(v, 0), len(cats[c]) - 1)))] for v in df[c]]
        return df

    return encode, decode, cats


def shap_top(sv, X_t, k=3):
    groups = [original_feature(c) for c in X_t.columns]
    vals = pd.DataFrame(sv.values[:, :, NF], columns=X_t.columns).T.groupby(groups).sum().T
    return [list(vals.iloc[i].abs().sort_values(ascending=False).index[:k]) for i in range(len(vals))]


def lime_top(exp, k=3):
    feats = []
    for idx, w in sorted(exp.as_map()[NF], key=lambda t: -abs(t[1])):
        f = FEATURES[idx]
        if f not in feats:
            feats.append(f)
        if len(feats) == k:
            break
    return feats


def readable_rule(rule: str) -> str:
    """'distance_to_town > 15823.05' -> 'Distance to town > 15.8 km'."""
    import re
    for f in sorted(FEATURES, key=len, reverse=True):
        if f in rule:
            if f.startswith("distance_"):
                rule = re.sub(r"(\d+\.?\d*)", lambda m: f"{float(m.group(1)) / 1000:.1f}", rule) + " km"
            rule = re.sub(r"(?<![<>])=", ": ", rule.replace(f, LABELS[f])).replace("<=", "≤").replace(">=", "≥")
            rule = re.sub(r"(\d+)\.00\b", r"\1", rule)
            break
    if rule.startswith("Install year not recorded ≤ 0"):
        return "Install year recorded"
    if rule.startswith("Install year not recorded"):
        return "Install year not recorded"
    return rule


def main():
    model = joblib.load(MODEL_PATH)
    train, test = split(load_modelling_data("mixed"))
    encode, decode, cats = encoder(train)
    explainer = LimeTabularExplainer(
        encode(train), feature_names=FEATURES, class_names=CLASSES, mode="classification",
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL_FEATURES],
        categorical_names={FEATURES.index(c): cats[c] for c in CATEGORICAL_FEATURES},
        discretize_continuous=True, random_state=RANDOM_STATE)
    predict = lambda arr: model.predict_proba(decode(arr))

    pts = test.sample(n=N_POINTS, random_state=RANDOM_STATE).reset_index(drop=True)
    sv, X_t = shap_values(model, pts[FEATURES])
    s_top = shap_top(sv, X_t)
    enc = encode(pts)
    l_exps = [explainer.explain_instance(enc[i], predict, labels=[NF], num_features=len(FEATURES),
                                         num_samples=2000) for i in range(len(pts))]
    l_top = [lime_top(e) for e in l_exps]

    overlap = [len(set(a) & set(b)) / 3 for a, b in zip(s_top, l_top)]
    top1 = [a[0] == b[0] for a, b in zip(s_top, l_top)]
    result = {"points_compared": N_POINTS, "mean_top3_overlap": round(float(np.mean(overlap)), 3),
              "top1_agreement": round(float(np.mean(top1)), 3),
              "share_with_at_least_2_of_3_shared": round(float(np.mean([o >= 2 / 3 for o in overlap])), 3),
              "most_common_shap_top1": pd.Series([a[0] for a in s_top]).value_counts().head(3).to_dict(),
              "most_common_lime_top1": pd.Series([b[0] for b in l_top]).value_counts().head(3).to_dict()}

    # Figure: LIME for the broken point with the highest predicted chance in this sample
    p = model.predict_proba(pts[FEATURES])[:, NF]
    nf = pts["y"].to_numpy() == NF
    i = int(np.argmax(np.where(nf, p, -1)))
    items = l_exps[i].as_list(label=NF)[:8][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh([readable_rule(t[0])[:52] for t in items], [t[1] for t in items],
            color=[plots.TERRA if t[1] > 0 else plots.NAVY for t in items], height=0.6)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("Effect on the chance of being broken (LIME weight)")
    ax.set_title(f"LIME explanation, broken point the model caught (chance broken {p[i]:.2f})\n"
                 f"SHAP top 3 for the same point: {', '.join(LABELS[f] for f in s_top[i])}", fontsize=9, loc="left")
    fig.savefig(FIG_DIR / "lime_local_caught.png")
    plt.close(fig)
    result["example"] = {"p_nf": round(float(p[i]), 3), "shap_top3": [LABELS[f] for f in s_top[i]],
                         "lime_top3": [LABELS[f] for f in l_top[i]]}
    (METRICS_DIR / "lime_shap_agreement.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
