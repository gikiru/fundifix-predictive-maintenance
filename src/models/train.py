"""
Module 4: train, tune, evaluate and register the water point status model.
Every run is logged to MLflow (tracking database: mlflow.db).

Run from repo root:  python -m src.models.train
View the log:        mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import json
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import skops.io as sio
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.models import plots
from src.models.config import (CATEGORICAL_FEATURES, CLASSES, CV_FOLDS, EXCLUDED, EXPERIMENT,
                               FEATURES, FIG_DIR, METRICS_DIR, MIXED_LABEL_SOURCES, MLFLOW_URI,
                               MODEL_DIR, MODEL_PATH, NUMERIC_FEATURES, RANDOM_STATE,
                               REGISTERED_MODEL)
from src.models.data import load_modelling_data, split, xy

warnings.filterwarnings("ignore", category=UserWarning)
NF = CLASSES.index("Non-Functional")


def preprocessor() -> ColumnTransformer:
    """Fitted inside each CV fold, so test data never informs the fill values or encoding."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)],
                             verbose_feature_names_out=False)


def candidates():
    """Model families, their search type and search space."""
    return {
        "baseline_majority": (DummyClassifier(strategy="most_frequent"), None, {}),
        "logistic_regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            "grid", {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                   random_state=RANDOM_STATE),
            "random", {"clf__n_estimators": [200, 400],
                       "clf__max_depth": [None, 10, 20],
                       "clf__min_samples_leaf": [1, 3, 5, 10],
                       "clf__max_features": ["sqrt", 0.3, 0.5]}),
        "xgboost": (
            XGBClassifier(objective="multi:softprob", tree_method="hist", eval_metric="mlogloss",
                          n_jobs=-1, random_state=RANDOM_STATE),
            "random", {"clf__n_estimators": [200, 400, 600],
                       "clf__max_depth": [3, 4, 6, 8],
                       "clf__learning_rate": [0.03, 0.05, 0.1],
                       "clf__subsample": [0.7, 0.85, 1.0],
                       "clf__colsample_bytree": [0.6, 0.8, 1.0],
                       "clf__min_child_weight": [1, 3, 5]}),
    }


def fit_params(name, y):
    """Class weighting for imbalance. XGBoost takes it as sample weights."""
    return {"clf__sample_weight": compute_sample_weight("balanced", y)} if name == "xgboost" else {}


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),   # = macro-averaged recall (G1)
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc_ovr_macro": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        **{f"recall_{c.replace(' ', '_').replace('-', '_').lower()}": r
           for c, r in zip(CLASSES, recall_score(y, pred, average=None, labels=[0, 1, 2], zero_division=0))},
        **{f"precision_{c.replace(' ', '_').replace('-', '_').lower()}": p
           for c, p in zip(CLASSES, precision_score(y, pred, average=None, labels=[0, 1, 2], zero_division=0))},
    }


def roc_data(model, X, y):
    proba = model.predict_proba(X)
    out = []
    for k, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve((y == k).astype(int), proba[:, k])
        out.append((c, fpr, tpr, roc_auc_score((y == k).astype(int), proba[:, k])))
    return out


def tune(name, estimator, search, space, X, y):
    pipe = Pipeline([("prep", preprocessor()), ("clf", estimator)])
    if search is None:
        pipe.fit(X, y)
        return pipe, {}, None
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    if search == "grid":
        s = GridSearchCV(pipe, space, scoring="f1_macro", cv=cv, n_jobs=-1)
    else:
        s = RandomizedSearchCV(pipe, space, n_iter=15, scoring="f1_macro", cv=cv, n_jobs=-1,
                               random_state=RANDOM_STATE)
    s.fit(X, y, **fit_params(name, y))
    return s.best_estimator_, s.best_params_, pd.DataFrame(s.cv_results_)


def main():
    for d in (FIG_DIR, METRICS_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = load_modelling_data("mixed")
    train, test = split(df)
    X_tr, y_tr = xy(train)
    X_te, y_te = xy(test)
    print(f"Mixed-label sources: {len(df):,} records | train {len(train):,} | test {len(test):,}")

    results, fitted = {}, {}
    for name, (est, search, space) in candidates().items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"data_scope": "mixed_label_sources", "family": name})
            mlflow.log_params({"train_rows": len(train), "test_rows": len(test),
                               "cv_folds": CV_FOLDS, "search": search or "none",
                               "features": ",".join(FEATURES), "class_weighting": "balanced"})
            model, best_params, cvr = tune(name, est, search, space, X_tr, y_tr)
            if best_params:
                mlflow.log_params({k.replace("clf__", "best_"): v for k, v in best_params.items()})
            if cvr is not None:
                best = cvr.loc[cvr["rank_test_score"] == 1].iloc[0]
                mlflow.log_metrics({"cv_macro_f1_mean": best["mean_test_score"],
                                    "cv_macro_f1_std": best["std_test_score"]})
                path = METRICS_DIR / f"cv_results_{name}.csv"
                cvr.to_csv(path, index=False)
                mlflow.log_artifact(str(path))
                for i, row in cvr.iterrows():             # one child run per tried setting
                    with mlflow.start_run(run_name=f"{name}_candidate_{i:02d}", nested=True):
                        mlflow.log_params({k.replace("param_clf__", ""): row[k]
                                           for k in cvr.columns if k.startswith("param_clf__")})
                        mlflow.log_metrics({"cv_macro_f1_mean": row["mean_test_score"],
                                            "cv_macro_f1_std": row["std_test_score"]})
            m = evaluate(model, X_te, y_te)
            mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})
            cm = confusion_matrix(y_te, model.predict(X_te), labels=[0, 1, 2])
            fig = FIG_DIR / f"confusion_{name}.png"
            plots.confusion(cm, CLASSES, f"Confusion matrix: {name.replace('_', ' ')} (test set)", fig)
            mlflow.log_artifact(str(fig))
            rep = METRICS_DIR / f"classification_report_{name}.txt"
            rep.write_text(classification_report(y_te, model.predict(X_te), target_names=CLASSES, digits=3))
            mlflow.log_artifact(str(rep))
            info = mlflow.sklearn.log_model(
                model, name="model",
                # skops is MLflow's safe format. We trust the types in a model we just trained.
                skops_trusted_types=sio.get_untrusted_types(data=sio.dumps(model)),
                signature=infer_signature(X_te.head(20), model.predict_proba(X_te.head(20))),
                input_example=X_te.head(3))
            results[name] = {"run_id": run.info.run_id, "model_uri": info.model_uri,
                             "best_params": best_params,
                             "cv_macro_f1": None if cvr is None else float(best["mean_test_score"]),
                             "test": m, "confusion_matrix": cm.tolist()}
            fitted[name] = model
            print(f"{name:20s} cv_f1={results[name]['cv_macro_f1']} test_f1={m['macro_f1']:.3f} "
                  f"bal_acc={m['balanced_accuracy']:.3f} nf_recall={m['recall_non_functional']:.3f}")

    # Choose the best model on cross-validation score, never on the test set
    tuned = {k: v for k, v in results.items() if v["cv_macro_f1"] is not None}
    best_name = max(tuned, key=lambda k: tuned[k]["cv_macro_f1"])
    best_model = fitted[best_name]
    print(f"Selected: {best_name}")

    plots.roc_curves(roc_data(best_model, X_te, y_te),
                     f"ROC curves, one class against the rest: {best_name.replace('_', ' ')}",
                     FIG_DIR / "roc_best_model.png")
    names = list(results)
    plots.model_comparison([n.replace("_", " ") for n in names],
                           [results[n]["test"]["macro_f1"] for n in names],
                           "Macro F1 on the test set", None, FIG_DIR / "model_comparison_f1.png")
    plots.model_comparison([n.replace("_", " ") for n in names],
                           [results[n]["test"]["balanced_accuracy"] for n in names],
                           "Balanced accuracy on the test set", 0.80,
                           FIG_DIR / "model_comparison_balanced_accuracy.png")

    # Register the selected model and give it the "champion" alias
    client = MlflowClient()
    mv = mlflow.register_model(results[best_name]["model_uri"], REGISTERED_MODEL)
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", mv.version)
    client.set_model_version_tag(REGISTERED_MODEL, mv.version, "stage", "Staging")
    client.set_model_version_tag(REGISTERED_MODEL, mv.version, "family", best_name)
    with mlflow.start_run(run_id=results[best_name]["run_id"]):
        mlflow.log_artifact(str(FIG_DIR / "roc_best_model.png"))
    joblib.dump(best_model, MODEL_PATH, compress=3)
    print(f"Registered {REGISTERED_MODEL} v{mv.version} (alias champion). Saved {MODEL_PATH}")

    # Operating point for Non-Functional: pick the lowest probability cut-off that reaches
    # 70% Non-Functional recall on out-of-fold training predictions, then check it on test.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(best_model, X_tr, y_tr, cv=cv, method="predict_proba",
                            params=fit_params(best_name, y_tr))[:, NF]
    grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    oof_recall = np.array([((oof >= t) & (y_tr == NF)).sum() / (y_tr == NF).sum() for t in grid])
    threshold = float(grid[oof_recall >= 0.70].max())
    p_te = best_model.predict_proba(X_te)[:, NF]
    flag = p_te >= threshold
    op = {"nf_threshold": threshold,
          "test_nf_recall": float((flag & (y_te == NF)).sum() / (y_te == NF).sum()),
          "test_nf_precision": float((flag & (y_te == NF)).sum() / max(flag.sum(), 1)),
          "test_share_flagged": float(flag.mean())}
    with mlflow.start_run(run_id=results[best_name]["run_id"]):
        mlflow.log_metrics({f"op_{k}": v for k, v in op.items()})
    results[best_name]["operating_point"] = op
    print(f"NF threshold {threshold:.2f}: test recall {op['test_nf_recall']:.3f}, "
          f"precision {op['test_nf_precision']:.3f}, flags {op['test_share_flagged']:.1%} of points")

    # Ablation (FO3): does adding location (latitude, longitude) change the result?
    with mlflow.start_run(run_name=f"ablation_{best_name}_with_location"):
        mlflow.set_tags({"data_scope": "mixed_label_sources", "purpose": "FO3 ablation"})
        num = NUMERIC_FEATURES + ["lat_deg", "lon_deg"]
        prep = ColumnTransformer([("num", preprocessor().transformers[0][1], num),
                                  ("cat", preprocessor().transformers[1][1], CATEGORICAL_FEATURES)])
        pipe = Pipeline([("prep", prep), ("clf", candidates()[best_name][0])])
        pipe.set_params(**results[best_name]["best_params"])
        cols = num + CATEGORICAL_FEATURES
        pipe.fit(train[cols], y_tr, **fit_params(best_name, y_tr))
        m_loc = evaluate(pipe, test[cols], y_te)
        mlflow.log_param("features", ",".join(cols))
        mlflow.log_metrics({f"test_{k}": v for k, v in m_loc.items()})
        results["ablation_with_location"] = {"test": m_loc}
        print(f"With location: test macro F1 {m_loc['macro_f1']:.3f} "
              f"(without: {results[best_name]['test']['macro_f1']:.3f})")

    # Comparison: same model settings trained on all 21,952 records (all 16 sources)
    all_df = load_modelling_data("all")
    a_tr, a_te = split(all_df)
    with mlflow.start_run(run_name="comparison_all_sources"):
        mlflow.set_tags({"data_scope": "all_sources", "family": best_name, "purpose": "comparison"})
        est = candidates()[best_name][0]
        pipe = Pipeline([("prep", preprocessor()), ("clf", est)])
        pipe.set_params(**results[best_name]["best_params"])
        Xa, ya = xy(a_tr)
        pipe.fit(Xa, ya, **fit_params(best_name, ya))
        m_all = evaluate(pipe, *xy(a_te))
        mixed_part = a_te[a_te["dataset_title"].isin(MIXED_LABEL_SOURCES)]
        m_mixed = evaluate(pipe, *xy(mixed_part))
        mlflow.log_params({"train_rows": len(a_tr), "test_rows": len(a_te)})
        mlflow.log_metrics({f"test_all_{k}": v for k, v in m_all.items()})
        mlflow.log_metrics({f"test_mixed_only_{k}": v for k, v in m_mixed.items()})
        results["comparison_all_sources"] = {"test_all": m_all, "test_mixed_only": m_mixed,
                                             "train_rows": len(a_tr), "test_rows": len(a_te),
                                             "mixed_test_rows": len(mixed_part)}
        print(f"All-sources model: test macro F1 {m_all['macro_f1']:.3f} on all sources, "
              f"{m_mixed['macro_f1']:.3f} on mixed-label sources only")

    # Metadata the API needs: which version, which threshold, which inputs
    meta = {"registered_model": REGISTERED_MODEL, "version": int(mv.version), "alias": "champion",
            "family": best_name, "classes": CLASSES, "features": FEATURES,
            "nf_threshold": results[best_name]["operating_point"]["nf_threshold"],
            "test_metrics": results[best_name]["test"],
            "training_scope": "4 mixed-label WPdx sources, 9,773 records"}
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))

    summary = {"selected_model": best_name, "registered_version": int(mv.version),
               "rows": {"mixed": len(df), "train": len(train), "test": len(test)},
               "class_counts_train": {c: int((y_tr == i).sum()) for i, c in enumerate(CLASSES)},
               "excluded_columns": EXCLUDED, "results": results}
    (METRICS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
