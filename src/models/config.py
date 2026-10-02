"""
Shared settings for Module 4 modelling.
Every script imports from here so the feature list, classes and paths stay in one place.
"""
from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

DB_PATH = Path("data/processed/fundifix.duckdb")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "fundifix_model.joblib"
FIG_DIR = Path("reports/figures")
METRICS_DIR = Path("reports/metrics")

MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT = "fundifix-status-model"
REGISTERED_MODEL = "fundifix-status-classifier"

# Target classes in a fixed order, so class index 2 is always Non-Functional
CLASSES = ["Functional", "Needs Repair", "Non-Functional"]

# The four sources with mixed labels (Module 3 bias check, finding F3).
# The other 12 sources label 100% of their records Non-Functional.
MIXED_LABEL_SOURCES = [
    "USAID_KIWASH_Kenya_2021",
    "Virtual Kenya_Busia, Kajiado, Kiambu, Kisumu_2012",
    "Virtual Kenya_Embu Kenya_2013-2014",
    "Marsabit Water_Kenya_2012",
]

NUMERIC_FEATURES = [
    "age_at_report",          # derived: report year minus install year
    "install_year_missing",   # 1 if install year was not recorded
    "local_population",
    "assigned_population",
    "usage_cap",
    "criticality",
    "pressure",
    "distance_to_primary",
    "distance_to_secondary",
    "distance_to_tertiary",
    "distance_to_city",
    "distance_to_town",
]
CATEGORICAL_FEATURES = [
    "water_source_clean",
    "water_tech_clean",
    "management_clean",
    "is_urban",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns kept out of the model, and why (reported in the validation report)
EXCLUDED = {
    "rehab_priority": "leakage: only filled for Non-Functional points",
    "would_gain_access": "leakage: only filled for Non-Functional points",
    "status_clean, status_id, prediction_*": "leakage: repeat or predict the target",
    "dataset_title, source": "data-source proxy (Module 3 finding F3)",
    "report_date, days_since_report, staleness": "reporting-time proxy for data source",
    "clean_adm1/2/3, lat_deg, lon_deg": "region identifiers, used for fairness checks instead (FO3)",
    "facility_type": "same value for every mixed-source record",
}

# Groups used for fairness analysis (Module 1 FO1, FO2)
GROUP_COLUMNS = ["clean_adm1", "is_urban", "road_access"]
