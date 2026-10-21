"""
Retraining with release gates. A retrained model is never put live automatically.

    python -m src.models.retrain                                  # refit on current data (scheduled check)
    python -m src.models.retrain --new-data data/new/inspections.csv   # add newly labelled records
    python -m src.models.retrain --promote 2 --approved-by "Name, role"  # human approval step

What it does
 1. Loads the mixed-label training data and, if given, new labelled records (for example
    FundiFix inspection results). New records are checked before use.
 2. Refits the champion's pipeline with the same settings, and sets a new visit threshold
    from out-of-fold predictions (the lowest cut-off that reaches 70% recall, as in Module 4).
 3. Compares challenger and champion on the same fixed benchmark test set, plus any new hold-out.
 4. Applies four release gates. All must pass:
      performance  balanced accuracy no more than 0.01 below the champion
      recall       visit list finds at least 65% of broken points
      fairness     county gap in share found no more than 5 points wider than the champion
      remote       points over 20 km from a main road are found at least as often as points under 5 km
 5. Logs everything to MLflow and registers the model. Alias "challenger" only if all gates pass.

Promotion to "champion" needs a named approver and is recorded in docs/model_change_log.md.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import skops.io as sio
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src.models.config import (CLASSES, CV_FOLDS, EXPERIMENT, FEATURES, MLFLOW_URI, MODEL_DIR, MODEL_PATH,
                               NUMERIC_FEATURES, RANDOM_STATE, REGISTERED_MODEL)
from src.models.data import add_derived_features, load_modelling_data, road_access_band, split
from src.models.train import evaluate
from src.monitoring.monitor import fairness, performance

NF = CLASSES.index("Non-Functional")
CANDIDATES = MODEL_DIR / "candidates"
REPORT_DIR = Path("reports/retraining")
CHANGE_LOG = Path("docs/model_change_log.md")
GATES = {"max_balanced_accuracy_drop": 0.01, "min_nf_recall": 0.65, "max_county_gap_rise": 0.05}
RAW_COLUMNS = ["install_year", "report_date", "local_population", "assigned_population", "usage_cap",
               "criticality", "pressure", "distance_to_primary", "distance_to_secondary", "distance_to_tertiary",
               "distance_to_city", "distance_to_town", "water_source_clean", "water_tech_clean",
               "management_clean", "is_urban", "clean_adm1", "target"]


def load_new_data(path: Path) -> pd.DataFrame:
    """Read and check newly labelled records. Raises ValueError with every problem found."""
    df = pd.read_csv(path)
    problems = [f"missing column: {c}" for c in RAW_COLUMNS if c not in df.columns]
    if problems:
        raise ValueError("; ".join(problems))
    bad_target = ~df["target"].isin(CLASSES)
    if bad_target.any():
        problems.append(f"{bad_target.sum()} rows with a status outside {CLASSES}")
    dist = [c for c in NUMERIC_FEATURES if c.startswith("distance_")]
    if (df[dist] < 0).any().any():
        problems.append("negative distances")
    if (df["install_year"] > datetime.now().year).any():
        problems.append("install year in the future")
    if df["target"].nunique() < 2:
        problems.append("all new records have the same status (the Module 3 source-bias pattern)")
    if problems:
        raise ValueError("; ".join(problems))
    df["is_urban"] = df["is_urban"].astype(str).str.lower().isin(["true", "1", "urban"])
    df = add_derived_features(df)
    df["road_access"] = road_access_band(df["distance_to_primary"])
    df["y"] = df["target"].map({c: i for i, c in enumerate(CLASSES)})
    df["dataset_title"] = "new_labelled_data"
    return df


def choose_threshold(model, X, y):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")[:, NF]
    grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    recall = np.array([((oof >= t) & (y == NF)).sum() / (y == NF).sum() for t in grid])
    return float(grid[recall >= 0.70].max())


def scorecard(model, thr, df):
    p = model.predict_proba(df[FEATURES])
    y = df["y"].to_numpy()
    perf = performance(y, p, thr)
    fair = fairness(df.reset_index(drop=True), y, p, thr)
    road = fair.get("road_access", {}).get("recall_by_group", {})
    return {**{k: round(float(v), 4) for k, v in perf.items()},
            "county_gap": round(fair.get("county", {}).get("recall_gap", float("nan")), 4),
            "recall_remote": road.get("over 20 km"), "recall_near": road.get("under 5 km"),
            "share_flagged": round(float((p[:, NF] >= thr).mean()), 4)}


def gates(ch, champ):
    return {
        "performance": ch["balanced_accuracy"] >= champ["balanced_accuracy"] - GATES["max_balanced_accuracy_drop"],
        "recall": ch["nf_recall"] >= GATES["min_nf_recall"],
        "fairness": ch["county_gap"] <= champ["county_gap"] + GATES["max_county_gap_rise"],
        "remote": (ch["recall_remote"] or 0) >= (ch["recall_near"] or 0),
    }


def retrain(new_data=None):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    champion = joblib.load(MODEL_PATH)
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
    train, bench = split(load_modelling_data("mixed"))          # same benchmark test set as Module 4
    new_test = None
    if new_data:
        new = load_new_data(Path(new_data))
        strat = new["y"] if new["y"].value_counts().min() >= 2 else None
        new_train, new_test = train_test_split(new, test_size=0.2, stratify=strat, random_state=RANDOM_STATE)
        train = pd.concat([train, new_train], ignore_index=True)

    challenger = clone(champion)
    challenger.fit(train[FEATURES], train["y"])
    thr = choose_threshold(champion, train[FEATURES], train["y"].to_numpy())

    result = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "champion_version": meta["version"], "train_rows": int(len(train)),
              "new_data": str(new_data) if new_data else None, "challenger_threshold": thr,
              "benchmark": {"champion": scorecard(champion, meta["nf_threshold"], bench),
                            "challenger": scorecard(challenger, thr, bench)}}
    if new_test is not None and len(new_test) >= 20:
        result["new_holdout"] = {"champion": scorecard(champion, meta["nf_threshold"], new_test),
                                 "challenger": scorecard(challenger, thr, new_test)}
    g = gates(result["benchmark"]["challenger"], result["benchmark"]["champion"])
    result["gates"] = g
    result["passed"] = all(g.values())

    with mlflow.start_run(run_name=f"retrain_{datetime.now():%Y%m%d_%H%M}") as run:
        mlflow.set_tags({"purpose": "retraining", "data_scope": "mixed_label_sources" + ("+new" if new_data else ""),
                         "gates": "pass" if result["passed"] else "fail"})
        mlflow.log_params({"train_rows": len(train), "nf_threshold": thr, "new_data": result["new_data"],
                           "champion_version": meta["version"]})
        mlflow.log_metrics({f"bench_{k}": v for k, v in result["benchmark"]["challenger"].items()
                            if isinstance(v, (int, float)) and v == v})
        mlflow.log_metrics({f"gate_{k}": int(v) for k, v in g.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in evaluate(challenger, bench[FEATURES], bench["y"]).items()})
        info = mlflow.sklearn.log_model(challenger, name="model",
                                        skops_trusted_types=sio.get_untrusted_types(data=sio.dumps(challenger)),
                                        signature=infer_signature(bench[FEATURES].head(20),
                                                                  challenger.predict_proba(bench[FEATURES].head(20))))
        result["run_id"] = run.info.run_id
    client = MlflowClient()
    mv = mlflow.register_model(info.model_uri, REGISTERED_MODEL)
    client.set_model_version_tag(REGISTERED_MODEL, mv.version, "gates", "pass" if result["passed"] else "fail")
    client.set_model_version_tag(REGISTERED_MODEL, mv.version, "nf_threshold", str(thr))
    if result["passed"]:
        client.set_registered_model_alias(REGISTERED_MODEL, "challenger", mv.version)
    result["registered_version"] = int(mv.version)

    CANDIDATES.mkdir(parents=True, exist_ok=True)
    joblib.dump(challenger, CANDIDATES / f"fundifix_model_v{mv.version}.joblib", compress=3)
    (CANDIDATES / f"fundifix_model_v{mv.version}.json").write_text(json.dumps(result, indent=2, default=str))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "latest.json").write_text(json.dumps(result, indent=2, default=str))

    c, h = result["benchmark"]["challenger"], result["benchmark"]["champion"]
    print(f"Champion v{meta['version']}: bal_acc={h['balanced_accuracy']:.3f} recall={h['nf_recall']:.3f} county_gap={h['county_gap']:.3f}")
    print(f"Challenger v{mv.version}: bal_acc={c['balanced_accuracy']:.3f} recall={c['nf_recall']:.3f} county_gap={c['county_gap']:.3f} threshold={thr:.2f}")
    print("Gates:", ", ".join(f"{k}={'pass' if v else 'FAIL'}" for k, v in g.items()))
    print(f"Registered v{mv.version} " + ("with alias 'challenger'. Awaiting human approval to promote."
                                          if result["passed"] else "without alias. Not eligible for promotion."))
    return result


def promote(version: int, approved_by: str):
    """Human step: make a gate-passing challenger the champion, keep the old one for rollback."""
    if not approved_by.strip():
        raise SystemExit("--approved-by is required")
    cand = CANDIDATES / f"fundifix_model_v{version}.joblib"
    info = json.loads((CANDIDATES / f"fundifix_model_v{version}.json").read_text())
    if not info["passed"]:
        raise SystemExit(f"Version {version} failed its release gates and cannot be promoted")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
    old = meta["version"]
    client.set_registered_model_alias(REGISTERED_MODEL, "previous_champion", old)
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", version)
    client.delete_registered_model_alias(REGISTERED_MODEL, "challenger")
    shutil.copy(MODEL_PATH, CANDIDATES / f"fundifix_model_v{old}_retired.joblib")
    shutil.copy(cand, MODEL_PATH)
    bench = info["benchmark"]["challenger"]
    meta.update({"version": int(version), "nf_threshold": info["challenger_threshold"],
                 "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "approved_by": approved_by, "benchmark": bench})
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    if not CHANGE_LOG.exists():
        CHANGE_LOG.write_text("# Model change log\n\n| Date (UTC) | From | To | Approved by | Balanced accuracy | Recall | County gap |\n|---|---|---|---|---|---|---|\n")
    with open(CHANGE_LOG, "a") as f:
        f.write(f"| {meta['promoted_at']} | v{old} | v{version} | {approved_by} | {bench['balanced_accuracy']:.3f} | "
                f"{bench['nf_recall']:.3f} | {bench['county_gap']:.3f} |\n")
    print(f"v{version} is now champion (v{old} kept as previous_champion). Next: "
          "python -m src.monitoring.monitor --set-baseline and python -m src.models.dashboard_data")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-data", help="CSV of newly labelled water points")
    ap.add_argument("--promote", type=int, help="registered version to make champion")
    ap.add_argument("--approved-by", default="", help="name and role of the person approving promotion")
    a = ap.parse_args()
    promote(a.promote, a.approved_by) if a.promote else retrain(a.new_data)
