"""
Load the Module 3 clean table and build the modelling dataset.
"""
import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.config import (CATEGORICAL_FEATURES, CLASSES, DB_PATH, FEATURES,
                               MIXED_LABEL_SOURCES, RANDOM_STATE, TEST_SIZE)


def add_derived_features(df: pd.DataFrame, report_year=None) -> pd.DataFrame:
    """Add age_at_report and clean types. Used in training and in the API."""
    df = df.copy()
    if report_year is None:
        report_year = pd.to_datetime(df["report_date"]).dt.year
    age = report_year - df["install_year"]
    df["age_at_report"] = age.where(age >= 0)          # negative ages are data errors
    df["install_year_missing"] = df["install_year"].isna().astype(int)
    df["is_urban"] = df["is_urban"].map({True: "urban", False: "rural"}).fillna("rural")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("object").fillna("Unknown").astype(str)
    return df


def road_access_band(distance_m: pd.Series) -> pd.Series:
    """Group distance to the nearest primary road into three bands (FO2)."""
    return pd.cut(distance_m, bins=[-np.inf, 5_000, 20_000, np.inf],
                  labels=["under 5 km", "5-20 km", "over 20 km"]).astype(str)


def load_modelling_data(scope: str = "mixed") -> pd.DataFrame:
    """scope='mixed' keeps the 4 mixed-label sources. scope='all' keeps every record."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM clean_wpdx").df()
    con.close()
    if scope == "mixed":
        df = df[df["dataset_title"].isin(MIXED_LABEL_SOURCES)]
    df = add_derived_features(df)
    df["road_access"] = road_access_band(df["distance_to_primary"])
    df["y"] = df["target"].map({c: i for i, c in enumerate(CLASSES)})
    return df.reset_index(drop=True)


def split(df: pd.DataFrame):
    """Stratified 80/20 hold-out split. Returns train and test frames."""
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["y"],
                                   random_state=RANDOM_STATE)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def xy(df: pd.DataFrame):
    return df[FEATURES], df["y"].to_numpy()
