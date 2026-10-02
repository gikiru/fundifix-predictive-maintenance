"""
Module 4 fairness analysis and mitigation (Fairlearn).

Decision being audited: "flag this water point for a visit" = P(Non-Functional) >= threshold.
Groups: county (FO1), urban/rural and distance to primary road (FO2), plus data source as a check.
Run from repo root after training:  python -m src.models.fairness
"""
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (MetricFrame, count, demographic_parity_difference,
                               demographic_parity_ratio, equalized_odds_difference,
                               false_positive_rate, selection_rate, true_positive_rate)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.models import plots
from src.models.config import CLASSES, CV_FOLDS, FEATURES, FIG_DIR, METRICS_DIR, MODEL_PATH, RANDOM_STATE
from src.models.data import load_modelling_data, split
from src.models.train import fit_params

warnings.filterwarnings("ignore")
NF = CLASSES.index("Non-Functional")
MIN_TEST_GROUP = 50          # smaller groups give unstable rates and are reported as "too few"
DI_FLAG = 0.80               # four-fifths rule for disparate impact
EO_FLAG = 0.10               # equalized odds difference above 10 points is flagged


class ScoreReader(BaseEstimator, ClassifierMixin):
    """Lets ThresholdOptimizer use pre-computed Non-Functional probabilities (column 'score')."""
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = X["score"].to_numpy()
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return (X["score"].to_numpy() >= 0.5).astype(int)


def group_table(y_true, y_flag, y3_true, y3_pred, groups):
    mf = MetricFrame(metrics={"records": count, "flag_rate": selection_rate,
                              "recall_nf": true_positive_rate, "false_alarm_rate": false_positive_rate},
                     y_true=y_true, y_pred=y_flag, sensitive_features=groups)
    t = mf.by_group
    t["actual_nf_share"] = pd.Series(y_true).groupby(groups.to_numpy()).mean()
    t["broken_points"] = pd.Series(y_true).groupby(groups.to_numpy()).sum()
    t["accuracy_3_class"] = pd.Series(y3_true == y3_pred).groupby(groups.to_numpy()).mean()
    return t


def summary_metrics(y_true, y_flag, groups):
    return {"demographic_parity_difference": demographic_parity_difference(y_true, y_flag, sensitive_features=groups),
            "disparate_impact_ratio": demographic_parity_ratio(y_true, y_flag, sensitive_features=groups),
            "equalized_odds_difference": equalized_odds_difference(y_true, y_flag, sensitive_features=groups)}


def main():
    model = joblib.load(MODEL_PATH)
    op = json.loads((METRICS_DIR / "training_summary.json").read_text())
    best = op["selected_model"]
    threshold = op["results"][best]["operating_point"]["nf_threshold"]

    train, test = split(load_modelling_data("mixed"))
    y_te3 = test["y"].to_numpy()
    p_te = model.predict_proba(test[FEATURES])[:, NF]
    y_true = (y_te3 == NF).astype(int)
    y_flag = (p_te >= threshold).astype(int)
    y_pred3 = model.predict(test[FEATURES])

    report, tables = {"threshold": threshold, "groupings": {}}, {}
    for col, label in [("clean_adm1", "county"), ("is_urban", "urban_rural"),
                       ("road_access", "road_access"), ("dataset_title", "data_source")]:
        g = test[col].astype(str)
        big = g.map(g.value_counts()) >= MIN_TEST_GROUP
        t = group_table(y_true[big], y_flag[big], y_te3[big], y_pred3[big], g[big])
        t.to_csv(METRICS_DIR / f"fairness_by_{label}.csv")
        s = summary_metrics(y_true[big], y_flag[big], g[big])
        s["groups_compared"] = int(big.groupby(g).first().sum())
        s["groups_too_small"] = sorted(g[~big].unique().tolist())
        s["accuracy_gap_3_class"] = float(t["accuracy_3_class"].max() - t["accuracy_3_class"].min())
        s["flag_di"] = bool(s["disparate_impact_ratio"] < DI_FLAG)
        s["flag_eo"] = bool(s["equalized_odds_difference"] > EO_FLAG)
        report["groupings"][label] = s
        tables[label] = t
        print(f"{label:12s} DI={s['disparate_impact_ratio']:.2f} EO={s['equalized_odds_difference']:.2f} "
              f"acc_gap={s['accuracy_gap_3_class']:.2f} groups={s['groups_compared']}")

    # Mitigation: equal-opportunity thresholds per county (FO1), fitted on out-of-fold
    # training scores so the test set stays untouched.
    target = "county"
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(clone(model), train[FEATURES], train["y"], cv=cv, method="predict_proba",
                            params=fit_params(best, train["y"].to_numpy()))[:, NF]
    g_tr = train["clean_adm1"].astype(str)
    keep_tr = g_tr.map(g_tr.value_counts()) >= MIN_TEST_GROUP * 4
    to = ThresholdOptimizer(estimator=ScoreReader().fit(None, None), constraints="true_positive_rate_parity",
                            objective="balanced_accuracy_score", prefit=True, predict_method="predict_proba")
    to.fit(pd.DataFrame({"score": oof[keep_tr]}), (train["y"].to_numpy()[keep_tr] == NF).astype(int),
           sensitive_features=g_tr[keep_tr])
    g_te = test["clean_adm1"].astype(str)
    keep_te = g_te.isin(g_tr[keep_tr].unique()) & (g_te.map(g_te.value_counts()) >= MIN_TEST_GROUP)
    y_mit = to.predict(pd.DataFrame({"score": p_te[keep_te]}), sensitive_features=g_te[keep_te],
                       random_state=RANDOM_STATE)
    before = summary_metrics(y_true[keep_te], y_flag[keep_te], g_te[keep_te])
    after = summary_metrics(y_true[keep_te], y_mit, g_te[keep_te])
    tb = group_table(y_true[keep_te], y_flag[keep_te], y_te3[keep_te], y_pred3[keep_te], g_te[keep_te])
    ta = group_table(y_true[keep_te], y_mit, y_te3[keep_te], y_pred3[keep_te], g_te[keep_te])
    overall = lambda yp: {"recall_nf": float(((yp == 1) & (y_true[keep_te] == 1)).sum() / y_true[keep_te].sum()),
                          "precision_nf": float(((yp == 1) & (y_true[keep_te] == 1)).sum() / max(yp.sum(), 1)),
                          "flag_rate": float(np.mean(yp))}
    report["mitigation"] = {"method": "Fairlearn ThresholdOptimizer, true positive rate parity by county",
                            "counties": sorted(g_te[keep_te].unique().tolist()),
                            "before": {**before, **overall(y_flag[keep_te])},
                            "after": {**after, **overall(y_mit)}}
    pd.concat({"before": tb, "after": ta}, axis=1).to_csv(METRICS_DIR / "fairness_mitigation_by_county.csv")
    print(f"Mitigation (county): EO {before['equalized_odds_difference']:.2f} -> {after['equalized_odds_difference']:.2f}, "
          f"recall {report['mitigation']['before']['recall_nf']:.2f} -> {report['mitigation']['after']['recall_nf']:.2f}")

    # Figure: recall of broken points by county, before and after mitigation
    order = tb.sort_values("recall_nf").index
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    yy = np.arange(len(order))
    ax.barh(yy - 0.2, tb.loc[order, "recall_nf"], height=0.38, color=plots.GREY, label="Before (one threshold)")
    ax.barh(yy + 0.2, ta.loc[order, "recall_nf"], height=0.38, color=plots.TEAL, label="After (per-county thresholds)")
    ax.set_yticks(yy, [f"{c} ({int(tb.loc[c, 'broken_points'])} broken)" for c in order])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of broken water points flagged (recall), test set")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Equal opportunity by county: before and after mitigation", fontsize=10, loc="left")
    fig.savefig(FIG_DIR / "fairness_county_recall.png")
    plt.close(fig)

    (METRICS_DIR / "fairness_report.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
