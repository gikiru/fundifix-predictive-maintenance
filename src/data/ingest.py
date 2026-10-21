"""
Ingestion: Load raw WPdx Kenya CSV files into DuckDB.
Runs as the first task in the Prefect pipeline.
"""
import hashlib
import json
import pandas as pd
import duckdb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
DB_PATH = Path("data/processed/fundifix.duckdb")
MANIFEST = RAW_DIR / "MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw_files(paths) -> bool:
    """Compare raw files with the fingerprints in data/raw/MANIFEST.json.
    Raw data is gitignored, so this is how the project proves it used the same download.
    A mismatch is logged, not fatal: a newer HDX download is allowed but must be noticed."""
    if not MANIFEST.exists():
        logger.warning("No raw data manifest found. Skipping integrity check.")
        return False
    expected = json.loads(MANIFEST.read_text())["files"]
    ok = True
    for p in paths:
        got = sha256(p)
        want = expected.get(p.name, {}).get("sha256")
        if got == want:
            logger.info(f"Integrity check passed: {p.name}")
        else:
            ok = False
            logger.warning(f"Integrity check FAILED: {p.name} differs from MANIFEST.json (new download?)")
    return ok


def load_raw(path: Path, skip_hxl: bool = True) -> pd.DataFrame:
    """Load a CSV, optionally skipping the HXL tag row (row index 1)."""
    df = pd.read_csv(path, skiprows=[1] if skip_hxl else [], low_memory=False)
    logger.info(f"Loaded {path.name}: {len(df):,} rows, {len(df.columns)} columns")
    return df


def ingest_to_duckdb(wpdx_path: Path, adm_path: Path, db_path: Path = DB_PATH) -> None:
    """Write both raw datasets to DuckDB as raw_wpdx and raw_adm tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    verify_raw_files([wpdx_path, adm_path])
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
