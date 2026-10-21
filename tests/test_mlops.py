"""
Tests for the Final Project MLOps pieces: retraining data checks and monitoring alerts.
Run from repo root: pytest tests/test_mlops.py -v
"""
import pandas as pd
import pytest

from src.models.retrain import RAW_COLUMNS, load_new_data
from src.monitoring.monitor import THRESHOLDS, alerts_for, level, status


def new_records(statuses):
    n = len(statuses)
    row = {"install_year": 2010, "report_date": "2026-03-01", "local_population": 500, "assigned_population": 300,
           "usage_cap": 500, "criticality": 0.5, "pressure": 0.6, "distance_to_primary": 9000,
           "distance_to_secondary": 3000, "distance_to_tertiary": 800, "distance_to_city": 40000,
           "distance_to_town": 7000, "water_source_clean": "Borehole/Tubewell",
           "water_tech_clean": "Hand Pump - Afridev", "management_clean": "Community Management",
           "is_urban": False, "clean_adm1": "Kitui"}
    return pd.DataFrame([{**row, "target": s} for s in statuses])


def test_new_data_with_one_status_is_rejected(tmp_path):
    """The Module 3 source-bias pattern (every point the same status) must block retraining."""
    p = tmp_path / "new.csv"
    new_records(["Non-Functional"] * 30).to_csv(p, index=False)
    with pytest.raises(ValueError, match="same status"):
        load_new_data(p)


def test_new_data_missing_column_is_rejected(tmp_path):
    p = tmp_path / "new.csv"
    new_records(["Functional", "Non-Functional"]).drop(columns=["target"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing column: target"):
        load_new_data(p)


def test_valid_new_data_is_prepared(tmp_path):
    p = tmp_path / "new.csv"
    new_records(["Functional", "Needs Repair", "Non-Functional"] * 5).to_csv(p, index=False)
    df = load_new_data(p)
    assert {"age_at_report", "road_access", "y"} <= set(df.columns)
    assert set(df["y"]) == {0, 1, 2}
    assert set(RAW_COLUMNS) <= set(df.columns)


def test_alert_levels():
    assert level(0.10, "drifted_feature_share") == "green"
    assert level(0.30, "drifted_feature_share") == "amber"
    assert level(0.60, "drifted_feature_share") == "red"
    assert level(0.55, "nf_recall", higher_is_worse=False) == "red"


def test_remote_points_losing_out_raises_red_alert():
    base = {"flag_rate": 0.41, "performance": {"nf_recall": 0.69, "hits_per_100": 27, "balanced_accuracy": 0.567},
            "fairness": {"road_access": {"recall_gap": 0.20}}}
    batch = {"drift": {"drifted_feature_share": 0.0}, "prediction": {"flag_rate": 0.41},
             "performance": {"nf_recall": 0.69, "hits_per_100": 27, "balanced_accuracy": 0.567},
             "fairness": {"road_access": {"recall_gap": 0.20,
                                          "recall_by_group": {"under 5 km": 0.70, "over 20 km": 0.50}}}}
    a = alerts_for(batch, base)
    assert status(a) == "red"
    assert any(x["area"] == "fairness" for x in a)
