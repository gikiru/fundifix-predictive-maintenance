"""
Monitoring job: data drift, prediction drift, performance and bias drift.

Run from repo root after training:
    python -m src.monitoring.monitor --set-baseline   # once, when a model version goes live
    python -m src.monitoring.monitor                   # every week (Prefect schedule or cron)

What it compares
    Reference data   the training split (what the model learned from)
    Baseline         the hold-out test results recorded when the model went live
                     (models/monitoring_baseline.json)
    Batches          1. hold-out test set (labelled): sanity check, should be green
                     2. WPdx records from the 12 sources not used in training (labels unreliable,
                        so drift only): the population the model would be asked to score
                     3. stress test (labelled, simulated): an older fleet, more unknown pump
                        types and longer road distances, to prove the alerts fire
                     4. live API requests from the explanation store, when there are at least 30

Tools: Evidently (data drift, HTML reports) and Fairlearn (fairness metrics by group).
Outputs: app/data/monitoring/summary.json (read by the dashboard Monitoring tab)
         app/data/monitoring/drift_<batch>.html (full Evidently reports)
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.models import explanation_store
from src.models.config import CATEGORICAL_FEATURES, CLASSES, FEATURES, MODEL_DIR, MODEL_PATH, NUMERIC_FEATURES
from src.models.data import load_modelling_data, road_access_band, split
from src.models.fairness import summary_metrics

NF = CLASSES.index("Non-Functional")
OUT_DIR = Path("app/data/monitoring")
BASELINE_PATH = MODEL_DIR / "monitoring_baseline.json"

# Alert thresholds (documented in docs/monitoring_plan.md). amber = review, red = act.
THRESHOLDS = {
    "drifted_feature_share": {"amber": 0.25, "red": 0.50},
    "flag_rate_change": {"amber": 0.10, "red": 0.20},           # absolute change in share flagged
    "nf_recall": {"amber": 0.65, "red": 0.60},                  # share of broken points found
    "hits_per_100": {"amber": 22, "red": 18},                   # broken points per 100 visits
    "balanced_accuracy_drop": {"amber": 0.03, "red": 0.05},
    "bias_drift": {"amber": 0.05, "red": 0.10},                 # rise in a fairness gap vs baseline
}
MIN_GROUP, MIN_BROKEN = 50, 10                                  # reliable fairness groups


def model_and_meta():
    return joblib.load(MODEL_PATH), json.loads((MODEL_DIR / "model_meta.json").read_text())


# ---------------- metrics ----------------

def performance(y3, p, thr):
    y_true = (y3 == NF).astype(int)
    flag = (p[:, NF] >= thr).astype(int)
    found = int((flag & y_true).sum())
    return {"balanced_accuracy": float(balanced_accuracy_score(y3, p.argmax(1))),
            "macro_f1": float(f1_score(y3, p.argmax(1), average="macro")),
            "nf_recall": found / max(int(y_true.sum()), 1),
            "hits_per_100": 100 * found / max(int(flag.sum()), 1),
            "random_hits_per_100": 100 * float(y_true.mean())}


def fairness(df, y3, p, thr):
    """Fairness of the visit flag. County uses reliable groups only (>= 50 records, >= 10 broken)."""
    y_true = pd.Series((y3 == NF).astype(int), index=df.index)
    flag = pd.Series((p[:, NF] >= thr).astype(int), index=df.index)
    out = {}
    for col, label in [("clean_adm1", "county"), ("road_access", "road_access"), ("is_urban", "urban_rural")]:
        g = df[col].astype(str)
        size, broken = g.map(g.value_counts()), g.map(y_true.groupby(g).sum())
        keep = (size >= MIN_GROUP) & (broken >= MIN_BROKEN)
        if keep.sum() == 0 or g[keep].nunique() < 2:
            continue
        s = summary_metrics(y_true[keep], flag[keep], g[keep])
        recall = (flag[keep] & y_true[keep]).groupby(g[keep]).sum() / y_true[keep].groupby(g[keep]).sum()
        out[label] = {"equalized_odds_difference": float(s["equalized_odds_difference"]),
                      "demographic_parity_difference": float(s["demographic_parity_difference"]),
                      "disparate_impact_ratio": float(s["disparate_impact_ratio"]),
                      "recall_gap": float(recall.max() - recall.min()),
                      "recall_by_group": {k: round(float(v), 4) for k, v in recall.items()},
                      "groups": int(g[keep].nunique())}
    return out


def evidently_drift(ref, cur, name):
    dd = DataDefinition(numerical_columns=NUMERIC_FEATURES, categorical_columns=CATEGORICAL_FEATURES)
    snap = Report([DataDriftPreset()]).run(current_data=Dataset.from_pandas(cur[FEATURES], data_definition=dd),
                                           reference_data=Dataset.from_pandas(ref[FEATURES], data_definition=dd))
    html = OUT_DIR / f"drift_{name}.html"
    snap.save_html(str(html))
    cols, share = {}, None
    for m in snap.dict()["metrics"]:
        cfg = m["config"]
        if cfg["type"].endswith("DriftedColumnsCount"):
            share = float(m["value"]["share"])
        elif cfg["type"].endswith("ValueDrift"):
            method, thr, v = cfg["method"], cfg["threshold"], float(m["value"])
            drifted = v < thr if "p_value" in method else v >= thr
            cols[cfg["column"]] = {"method": method, "score": round(v, 4), "threshold": thr, "drifted": bool(drifted)}
    return {"drifted_feature_share": share, "features": cols, "html": html.name}


def level(value, rule, higher_is_worse=True):
    t = THRESHOLDS[rule]
    if value is None:
        return "green"
    if higher_is_worse:
        return "red" if value >= t["red"] else "amber" if value >= t["amber"] else "green"
    return "red" if value < t["red"] else "amber" if value < t["amber"] else "green"


def alerts_for(batch, baseline):
    a = []
    def add(sev, area, msg):
        if sev != "green":
            a.append({"severity": sev, "area": area, "message": msg})
    d = batch["drift"]["drifted_feature_share"]
    add(level(d, "drifted_feature_share"), "data drift", f"{d:.0%} of inputs have drifted from the training data")
    ch = abs(batch["prediction"]["flag_rate"] - baseline["flag_rate"])
    add(level(ch, "flag_rate_change"), "prediction drift",
        f"Share flagged is {batch['prediction']['flag_rate']:.0%} against {baseline['flag_rate']:.0%} at go-live")
    perf = batch.get("performance")
    if perf:
        add(level(perf["nf_recall"], "nf_recall", False), "performance",
            f"Visit list finds {perf['nf_recall']:.0%} of broken points (go-live {baseline['performance']['nf_recall']:.0%})")
        add(level(perf["hits_per_100"], "hits_per_100", False), "performance",
            f"{perf['hits_per_100']:.0f} broken points per 100 visits (go-live {baseline['performance']['hits_per_100']:.0f})")
        drop = baseline["performance"]["balanced_accuracy"] - perf["balanced_accuracy"]
        add(level(drop, "balanced_accuracy_drop"), "performance", f"Balanced accuracy fell by {drop:.3f}")
    for grp, f in (batch.get("fairness") or {}).items():
        b = baseline["fairness"].get(grp)
        if not b:
            continue
        rise = f["recall_gap"] - b["recall_gap"]
        add(level(rise, "bias_drift"), "bias drift",
            f"{grp.replace('_', ' ')}: gap in share of broken points found is {f['recall_gap']:.0%} (go-live {b['recall_gap']:.0%})")
        if grp == "road_access":
            r = f["recall_by_group"]
            if r.get("over 20 km", 1) < r.get("under 5 km", 0):
                add("red", "fairness", "Remote points (over 20 km from a main road) are now found less often than nearby points")
    return a


def status(alerts):
    s = {x["severity"] for x in alerts}
    return "red" if "red" in s else "amber" if "amber" in s else "green"


# ---------------- batches ----------------

def stress_test(test, seed=7):
    """Simulated degradation of inputs. Labels are unchanged, so any drop shows sensitivity."""
    rng = np.random.default_rng(seed)
    s = test.copy()
    s["age_at_report"] = s["age_at_report"] + 8
    unknown = rng.random(len(s)) < 0.30
    s.loc[unknown, "water_tech_clean"] = "Unknown"
    s["distance_to_primary"] = s["distance_to_primary"] * 1.5
    s["road_access"] = road_access_band(s["distance_to_primary"])
    return s


def describe(df, name, label, model, thr, ref, labelled):
    p = model.predict_proba(df[FEATURES])
    b = {"name": name, "label": label, "rows": int(len(df)), "labelled": labelled,
         "drift": evidently_drift(ref, df, name),
         "prediction": {"flag_rate": float((p[:, NF] >= thr).mean()), "mean_p_nf": float(p[:, NF].mean())}}
    if labelled:
        y3 = df["y"].to_numpy()
        b["performance"] = performance(y3, p, thr)
        b["fairness"] = fairness(df.reset_index(drop=True), y3, p, thr)
    return b


def main(set_baseline=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, meta = model_and_meta()
    thr = meta["nf_threshold"]
    train, test = split(load_modelling_data("mixed"))

    if set_baseline or not BASELINE_PATH.exists():
        p = model.predict_proba(test[FEATURES])
        base = {"model_version": meta["version"], "set_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "nf_threshold": thr, "flag_rate": float((p[:, NF] >= thr).mean()),
                "performance": performance(test["y"].to_numpy(), p, thr),
                "fairness": fairness(test, test["y"].to_numpy(), p, thr)}
        BASELINE_PATH.write_text(json.dumps(base, indent=2))
        print(f"Baseline written for model version {meta['version']}")
    baseline = json.loads(BASELINE_PATH.read_text())

    allsrc = load_modelling_data("all")
    outside = allsrc[~allsrc["dataset_title"].isin(set(train["dataset_title"]) | set(test["dataset_title"]))]
    batches = [
        describe(test, "holdout", "Hold-out test set (labelled)", model, thr, train, True),
        describe(outside, "outside_sources", "WPdx points from sources not used in training (drift only)",
                 model, thr, train, False),
        describe(stress_test(test), "stress_test", "Stress test: simulated older, less documented fleet",
                 model, thr, train, True),
    ]
    live = explanation_store.load_features()
    if len(live) >= 30:
        batches.append(describe(live, "api_requests", f"Live API requests ({len(live)})", model, thr, train, False))

    for b in batches:
        b["alerts"] = alerts_for(b, baseline)
        b["status"] = status(b["alerts"])
        print(f"{b['label']:60s} rows={b['rows']:>6,} drift={b['drift']['drifted_feature_share']:.0%} "
              f"flagged={b['prediction']['flag_rate']:.0%} status={b['status'].upper()}")

    summary = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "model_version": meta["version"], "nf_threshold": thr, "thresholds": THRESHOLDS,
               "baseline": baseline, "reference_rows": int(len(train)),
               "explanation_store": explanation_store.summary(), "batches": batches}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    # History: one line per run, so trends can be charted over time
    hist = OUT_DIR / "history.jsonl"
    with open(hist, "a") as f:
        f.write(json.dumps({"run_at": summary["run_at"], "model_version": meta["version"],
                            **{b["name"]: {"status": b["status"], "drift": b["drift"]["drifted_feature_share"],
                                           "flag_rate": b["prediction"]["flag_rate"],
                                           "nf_recall": (b.get("performance") or {}).get("nf_recall")}
                               for b in batches}}) + "\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-baseline", action="store_true", help="record go-live results for the current model")
    main(ap.parse_args().set_baseline)
