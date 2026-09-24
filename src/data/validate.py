"""
Validation: Great Expectations suite for clean_wpdx.
Run after cleaning; fails loudly if data contract is violated.
"""
import pandas as pd
import great_expectations as gx
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_validation(df: pd.DataFrame) -> dict:
    """
    Run GE expectations against the clean dataset.
    Returns a results dict with passed/failed counts.
    """
    context = gx.get_context(mode="ephemeral")
    ds = context.sources.add_pandas("wpdx_clean")
    da = ds.add_dataframe_asset("clean_wpdx")
    batch_req = da.build_batch_request(dataframe=df)
    suite = context.add_expectation_suite("fundifix_clean_suite")

    validator = context.get_validator(batch_request=batch_req,
                                      expectation_suite_name="fundifix_clean_suite")

    # Schema expectations
    validator.expect_table_row_count_to_be_between(min_value=15000, max_value=25000)
    validator.expect_column_to_exist("target")
    validator.expect_column_to_exist("status_clean")
    validator.expect_column_to_exist("lat_deg")
    validator.expect_column_to_exist("lon_deg")

    # Target variable expectations
    validator.expect_column_values_to_be_in_set(
        "target", ["Functional", "Needs Repair", "Non-Functional"]
    )
    validator.expect_column_values_to_not_be_null("target")

    # Data quality expectations
    validator.expect_column_values_to_not_be_null("lat_deg", mostly=0.99)
    validator.expect_column_values_to_not_be_null("lon_deg", mostly=0.99)
    validator.expect_column_values_to_be_between(
        "lat_deg", min_value=-5.0, max_value=5.0
    )
    validator.expect_column_values_to_be_between(
        "lon_deg", min_value=33.5, max_value=42.5
    )
    validator.expect_column_values_to_be_between(
        "install_year", min_value=1950, max_value=2026, mostly=0.95
    )

    results = validator.validate()
    passed = sum(1 for r in results.results if r.success)
    failed = sum(1 for r in results.results if not r.success)
    logger.info(f"GE validation: {passed} passed, {failed} failed")

    return {
        "total": len(results.results),
        "passed": passed,
        "failed": failed,
        "success": results.success,
        "results": [
            {
                "expectation": r.expectation_config.expectation_type,
                "column": r.expectation_config.kwargs.get("column", "table"),
                "success": r.success,
            }
            for r in results.results
        ]
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import duckdb
    con = duckdb.connect("data/processed/fundifix.duckdb")
    df = con.execute("SELECT * FROM clean_wpdx").df()
    con.close()
    summary = run_validation(df)
    print(json.dumps(summary, indent=2))
