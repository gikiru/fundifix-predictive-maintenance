"""
Transformation: Feature engineering on clean_wpdx → features_wpdx.
"""
import pandas as pd
import duckdb
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
DB_PATH = Path("data/processed/fundifix.duckdb")

CATEGORICAL_FEATURES = [
    "water_source_clean", "water_source_category",
    "water_tech_clean", "water_tech_category",
    "facility_type", "management_clean",
    "clean_adm1", "clean_adm2", "clean_adm3",
    "is_urban",
]

NUMERIC_FEATURES = [
    "lat_deg", "lon_deg",
    "local_population", "assigned_population",
    "usage_cap", "criticality", "pressure",
    "distance_to_primary", "distance_to_secondary",
    "distance_to_tertiary", "distance_to_city", "distance_to_town",
    "days_since_report", "staleness",
    "install_year", "install_year_missing",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features and encode categoricals."""

    # 1. Water point age (years since installation)
    if "install_year" in df.columns:
        df["wp_age_years"] = 2026 - df["install_year"]
        df["wp_age_years"] = df["wp_age_years"].clip(lower=0)

    # 2. Median-impute numerics (flag already captured in install_year_missing)
    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # 3. Frequency-encode high-cardinality categoricals
    for col in ["clean_adm2", "clean_adm3", "management_clean"]:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq).fillna(0)

    # 4. One-hot encode lower-cardinality categoricals
    low_card = ["water_source_category", "water_tech_category",
                "facility_type", "is_urban", "clean_adm1"]
    for col in low_card:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])

    logger.info(f"Feature matrix: {len(df):,} rows x {len(df.columns)} columns")
    return df


def run_transform(db_path: Path = DB_PATH) -> pd.DataFrame:
    con = duckdb.connect(str(db_path))
    clean = con.execute("SELECT * FROM clean_wpdx").df()
    feat = engineer_features(clean)
    con.execute("DROP TABLE IF EXISTS features_wpdx")
    con.register("feat_df", feat)
    con.execute("CREATE TABLE features_wpdx AS SELECT * FROM feat_df")
    logger.info(f"Saved features_wpdx to DuckDB")
    con.close()
    return feat


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_transform()
