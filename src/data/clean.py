"""
Cleaning: Apply documented cleaning rules to raw_wpdx and produce clean_wpdx.
All decisions trace to Module 1 Section 1.3 and Module 2 Data Dictionary.
"""
import pandas as pd
import duckdb
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path("data/processed/fundifix.duckdb")

# Columns dropped due to >80% missingness or confirmed empty (Module 2 Data Dictionary)
DROP_COLS = [
    "installer", "pay_clean", "rehab_year", "rehabilitator",
    "fecal_coliform_presence", "fecal_coliform_value",
    "predicted_status_0y", "predicted_status_2y",
    "prediction_yes_0y", "prediction_no_0y",
    "predicted_category", "clean_adm4", "scheme_id",
    "subjective_quality", "status_id",   # status_id is NOT the target variable
    "notes",   # water point names, some after individuals (personal data)
]

# Target variable mapping (Module 1 Section 1.3 decision)
FUNCTIONAL     = ["Functional", "Functional, not in use"]
NEEDS_REPAIR   = ["Functional, needs repair"]
NON_FUNCTIONAL = ["Non-Functional", "Non-Functional, dry season", "Abandoned/Decommissioned"]

STATUS_MAP = (
    {s: "Functional"     for s in FUNCTIONAL} |
    {s: "Needs Repair"   for s in NEEDS_REPAIR} |
    {s: "Non-Functional" for s in NON_FUNCTIONAL}
)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and return a clean DataFrame."""
    n_start = len(df)

    # 1. Drop excluded columns (ignore missing ones gracefully)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 2. Remove erroneous install_year values (future dates)
    if "install_year" in df.columns:
        bad = df["install_year"] > 2026
        logger.info(f"Removing {bad.sum()} records with install_year > 2026")
        df = df[~bad].copy()

    # 3. Map status_clean to three-class target
    df["target"] = df["status_clean"].map(STATUS_MAP)
    unmapped = df["target"].isna().sum()
    if unmapped > 0:
        logger.warning(f"{unmapped} records have unmapped status_clean values — dropping")
        df = df[df["target"].notna()].copy()

    # 4. Add missingness flag for install_year (informative missingness)
    if "install_year" in df.columns:
        df["install_year_missing"] = df["install_year"].isna().astype(int)

    # 5. Drop exact duplicate rows
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Dropped {n_before - len(df)} exact duplicates")

    logger.info(f"Clean dataset: {n_start:,} → {len(df):,} rows")
    return df


def run_cleaning(db_path: Path = DB_PATH) -> pd.DataFrame:
    con = duckdb.connect(str(db_path))
    raw = con.execute("SELECT * FROM raw_wpdx").df()
    clean_df = clean(raw)
    con.execute("DROP TABLE IF EXISTS clean_wpdx")
    con.register("clean_df", clean_df)
    con.execute("CREATE TABLE clean_wpdx AS SELECT * FROM clean_df")
    logger.info(f"Saved clean_wpdx to DuckDB ({len(clean_df):,} rows)")
    con.close()
    return clean_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_cleaning()
