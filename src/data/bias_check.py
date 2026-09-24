"""
Bias Detection: check for representation bias in clean_wpdx using Fairlearn.
Runs before modeling. Checks two things per group (county, urban/rural):
  1. How many records each group has (representation)
  2. The share of Non-Functional water points in each group (label balance)
Fairness Objectives FO1 and FO2 from Module 1.
"""
import pandas as pd
import logging
import json
from fairlearn.metrics import MetricFrame, count, selection_rate

logger = logging.getLogger(__name__)

MIN_GROUP_SIZE = 100      # groups smaller than this give unreliable rates
DISPARITY_FLAG = 0.20     # flag a gap in Non-Functional share above 20 points


def _group_report(df: pd.DataFrame, group_col: str) -> MetricFrame:
    """Non-Functional share and record count per group, via Fairlearn MetricFrame."""
    y = (df["target"] == "Non-Functional").astype(int)
    return MetricFrame(
        metrics={"records": count, "non_functional_share": selection_rate},
        y_true=y,
        y_pred=y,
        sensitive_features=df[group_col].astype(str),
    )


def check_representation_bias(df: pd.DataFrame) -> dict:
    results = {
        "overall_class_distribution":
            df["target"].value_counts(normalize=True).round(4).to_dict()
    }

    # County (FO1): only compare counties with enough records
    if "clean_adm1" in df.columns:
        mf = _group_report(df, "clean_adm1").by_group
        large = mf[mf["records"] >= MIN_GROUP_SIZE]
        share = large["non_functional_share"]
        gap = round(float(share.max() - share.min()), 4) if len(large) > 1 else 0.0
        results["county_representation_bias"] = {
            "counties_total": int(len(mf)),
            "counties_with_100plus_records": int(len(large)),
            "counties_below_threshold": int((mf["records"] < MIN_GROUP_SIZE).sum()),
            "highest_county": share.idxmax() if len(large) else None,
            "highest_share": round(float(share.max()), 4) if len(large) else None,
            "lowest_county": share.idxmin() if len(large) else None,
            "lowest_share": round(float(share.min()), 4) if len(large) else None,
            "disparity": gap,
            "flag": bool(gap > DISPARITY_FLAG),
        }
        logger.info(f"County gap (>= {MIN_GROUP_SIZE} records): {gap:.3f}")

    # Urban/rural (FO2)
    if "is_urban" in df.columns:
        mf = _group_report(df, "is_urban").by_group
        share = mf["non_functional_share"]
        gap = round(float(share.max() - share.min()), 4) if len(mf) > 1 else 0.0
        results["urban_rural_bias"] = {
            "records": {k: int(v) for k, v in mf["records"].items()},
            "non_functional_share": {k: round(float(v), 4) for k, v in share.items()},
            "disparity": gap,
            "flag": bool(gap > DISPARITY_FLAG),
        }
        logger.info(f"Urban/rural gap: {gap:.3f}")

    # Data source: does the label depend on which organisation collected the record?
    if "dataset_title" in df.columns:
        mf = _group_report(df, "dataset_title").by_group
        share = mf["non_functional_share"]
        all_nf = mf[share == 1.0]
        results["source_bias"] = {
            "sources_total": int(len(mf)),
            "sources_reporting_100pct_non_functional": int(len(all_nf)),
            "records_from_those_sources": int(all_nf["records"].sum()),
            "flag": bool(len(all_nf) > 0),
        }
        logger.info(f"Sources with 100% Non-Functional labels: {len(all_nf)}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import duckdb
    con = duckdb.connect("data/processed/fundifix.duckdb")
    df = con.execute("SELECT * FROM clean_wpdx").df()
    con.close()
    print(json.dumps(check_representation_bias(df), indent=2, default=str))
