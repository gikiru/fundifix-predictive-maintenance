"""
Ingestion: Load raw WPdx Kenya CSV files into DuckDB.
Runs as the first task in the Prefect pipeline.
"""
import pandas as pd
import duckdb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
DB_PATH = Path("data/processed/fundifix.duckdb")


def load_raw(path: Path, skip_hxl: bool = True) -> pd.DataFrame:
    """Load a CSV, optionally skipping the HXL tag row (row index 1)."""
    df = pd.read_csv(path, skiprows=[1] if skip_hxl else [], low_memory=False)
    logger.info(f"Loaded {path.name}: {len(df):,} rows, {len(df.columns)} columns")
    return df


def ingest_to_duckdb(wpdx_path: Path, adm_path: Path, db_path: Path = DB_PATH) -> None:
    """Write both raw datasets to DuckDB as raw_wpdx and raw_adm tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    wpdx = load_raw(wpdx_path)
    adm = load_raw(adm_path)
    con = duckdb.connect(str(db_path))
    con.execute("DROP TABLE IF EXISTS raw_wpdx")
    con.execute("DROP TABLE IF EXISTS raw_adm")
    con.register("wpdx_df", wpdx)
    con.register("adm_df", adm)
    con.execute("CREATE TABLE raw_wpdx AS SELECT * FROM wpdx_df")
    con.execute("CREATE TABLE raw_adm AS SELECT * FROM adm_df")
    logger.info(f"Written to DuckDB: raw_wpdx ({len(wpdx):,} rows), raw_adm ({len(adm):,} rows)")
    con.close()


if __name__ == "__main__":
    ingest_to_duckdb(RAW_DIR / "wpdx_enhanced.csv", RAW_DIR / "adm_analysis.csv")
