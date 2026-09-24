"""
Prefect DAG: FundiFix Predictive Maintenance Data Pipeline.
Orchestrates ingestion → cleaning → transformation → validation → bias check → anonymization.
Run: python src/data/pipeline.py
"""
from prefect import flow, task, get_run_logger
from pathlib import Path
import pandas as pd
import duckdb

# Local imports (run from repo root)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.ingest import ingest_to_duckdb
from src.data.clean import run_cleaning
from src.data.transform import run_transform
from src.data.validate import run_validation
from src.data.bias_check import check_representation_bias
from src.data.anonymize import anonymize_for_export
from src.data.audit_log import log_event

RAW_DIR = Path("data/raw")
DB_PATH = Path("data/processed/fundifix.duckdb")


@task(name="ingest-raw-data")
def task_ingest():
    logger = get_run_logger()
    logger.info("Ingesting raw WPdx Kenya data")
    ingest_to_duckdb(
        RAW_DIR / "wpdx_enhanced.csv",
        RAW_DIR / "adm_analysis.csv",
        DB_PATH,
    )
    log_event("INGEST", "wpdx_enhanced.csv + adm_analysis.csv",
              {"source": "OCHA HDX", "db": str(DB_PATH)})


@task(name="clean-data")
def task_clean() -> int:
    logger = get_run_logger()
    logger.info("Cleaning raw_wpdx")
    df = run_cleaning(DB_PATH)
    log_event("CLEAN", "raw_wpdx",
              {"rows_out": len(df), "target_classes": df["target"].value_counts().to_dict()})
    return len(df)


@task(name="transform-features")
def task_transform():
    logger = get_run_logger()
    logger.info("Engineering features")
    df = run_transform(DB_PATH)
    log_event("TRANSFORM", "clean_wpdx", {"features_out": len(df.columns)})
    return len(df.columns)


@task(name="validate-data")
def task_validate() -> bool:
    logger = get_run_logger()
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM clean_wpdx").df()
    con.close()
    summary = run_validation(df)
    log_event("VALIDATE", "clean_wpdx", summary)
    if not summary["success"]:
        logger.warning(f"Validation FAILED: {summary['failed']} expectations not met")
    else:
        logger.info(f"Validation PASSED: {summary['passed']}/{summary['total']}")
    return summary["success"]


@task(name="bias-detection")
def task_bias():
    logger = get_run_logger()
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM clean_wpdx").df()
    con.close()
    results = check_representation_bias(df)
    log_event("BIAS_CHECK", "clean_wpdx", results)
    flagged = any(
        v.get("flag", False)
        for v in results.values()
        if isinstance(v, dict)
    )
    if flagged:
        logger.warning("Bias check: one or more disparities FLAGGED — review before modeling")
    else:
        logger.info("Bias check: all disparities within acceptable range")
    return results


@task(name="anonymize-export")
def task_anonymize():
    logger = get_run_logger()
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM clean_wpdx").df()
    con.close()
    anon = anonymize_for_export(df)
    out = Path("data/processed/wpdx_anonymised_export.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    anon.to_csv(out, index=False)
    log_event("EXPORT", str(out),
              {"rows": len(anon), "columns": len(anon.columns),
               "removed": [c for c in df.columns if c not in anon.columns],
               "personal_name_fields_removed_at_clean": ["notes"]})
    logger.info(f"Anonymised export saved: {out}")


@flow(name="fundifix-data-pipeline", log_prints=True)
def pipeline():
    """Main pipeline flow: ingest → clean → transform → validate → bias → anonymize."""
    task_ingest()
    n_rows = task_clean()
    n_features = task_transform()
    valid = task_validate()
    task_bias()
    task_anonymize()
    print(f"Pipeline complete: {n_rows:,} rows, {n_features} features, validation={'PASS' if valid else 'WARN'}")


if __name__ == "__main__":
    pipeline()
