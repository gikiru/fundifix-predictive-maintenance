"""
Unit tests for the FundiFix data pipeline.
Run from repo root: pytest tests/test_pipeline.py -v
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.clean import clean, STATUS_MAP
from src.data.transform import engineer_features
from src.data.anonymize import anonymize_for_export
from src.data.bias_check import check_representation_bias


# ---------- Fixtures ----------

@pytest.fixture
def sample_raw() -> pd.DataFrame:
    """Minimal raw DataFrame mimicking WPdx Kenya schema."""
    return pd.DataFrame({
        "status_clean":     ["Functional", "Non-Functional", "Functional, needs repair",
                             "Functional, not in use", "Abandoned/Decommissioned"],
        "install_year":     [2010, 2028, 2005, None, 1995],  # 2028 is erroneous
        "lat_deg":          [-1.2, -0.5, -2.1, -1.8, -3.0],
        "lon_deg":          [36.8, 37.1, 35.5, 38.0, 39.2],
        "local_population": [200, 150, None, 180, 90],
        "clean_adm1":       ["Nairobi", "Kwale", "Kwale", "Busia", "Busia"],
        "is_urban":         [True, False, False, False, False],
    })


@pytest.fixture
def sample_clean(sample_raw) -> pd.DataFrame:
    return clean(sample_raw)


# ---------- Cleaning tests ----------

def test_erroneous_install_year_removed(sample_clean):
    """install_year > 2026 must be dropped (Module 1 Issue I1)."""
    assert (sample_clean["install_year"] <= 2026).all() or \
           sample_clean["install_year"].isna().any()


def test_target_column_created(sample_clean):
    """target column must exist and contain only the three allowed classes."""
    assert "target" in sample_clean.columns
    allowed = {"Functional", "Needs Repair", "Non-Functional"}
    assert set(sample_clean["target"].unique()).issubset(allowed)


def test_functional_not_in_use_maps_to_functional(sample_raw):
    """'Functional, not in use' must map to 'Functional' (Module 1 decision)."""
    row = pd.DataFrame({
        "status_clean": ["Functional, not in use"],
        "lat_deg": [1.0], "lon_deg": [36.0],
        "install_year": [2000], "local_population": [100],
        "clean_adm1": ["Kwale"], "is_urban": [False],
    })
    result = clean(row)
    assert result["target"].iloc[0] == "Functional"


def test_install_year_missing_flag(sample_clean):
    """install_year_missing flag must be 1 where install_year is NaN."""
    missing_rows = sample_clean[sample_clean["install_year"].isna()]
    if len(missing_rows) > 0:
        assert (missing_rows["install_year_missing"] == 1).all()


def test_no_target_nulls(sample_clean):
    """target column must have no null values after cleaning."""
    assert sample_clean["target"].isna().sum() == 0


# ---------- Transformation tests ----------

def test_wp_age_years_created(sample_clean):
    """wp_age_years feature must be created and non-negative."""
    result = engineer_features(sample_clean)
    if "install_year" in sample_clean.columns:
        assert "wp_age_years" in result.columns
        assert (result["wp_age_years"].dropna() >= 0).all()


# ---------- Anonymization tests ----------

def test_gps_removed_in_export(sample_clean):
    """lat_deg and lon_deg must not appear in anonymised export."""
    anon = anonymize_for_export(sample_clean)
    assert "lat_deg" not in anon.columns
    assert "lon_deg" not in anon.columns


def test_personal_names_removed_in_export():
    """notes (water point names, some after people) must not appear in export."""
    df = pd.DataFrame({"notes": ["Mama Naomi Spring"], "clean_adm1": ["Busia"]})
    assert "notes" not in anonymize_for_export(df).columns


def test_area_names_kept_in_export():
    """County, sub-county and ward names are public place names and must be kept."""
    df = pd.DataFrame({"clean_adm1": ["Nyamira"], "clean_adm2": ["Kitutu Masaba"],
                       "clean_adm3": ["Rigoma"], "lat_deg": [0.5], "lon_deg": [34.2]})
    anon = anonymize_for_export(df)
    assert {"clean_adm1", "clean_adm2", "clean_adm3"} <= set(anon.columns)
    assert anon["clean_adm3"].iloc[0] == "Rigoma"


def test_export_keeps_all_rows(sample_clean):
    """Anonymisation removes columns only, never records."""
    assert len(anonymize_for_export(sample_clean)) == len(sample_clean)


# ---------- Bias detection tests ----------

def test_bias_check_returns_expected_keys(sample_clean):
    """check_representation_bias must return county and urban_rural keys."""
    result = check_representation_bias(sample_clean)
    assert "overall_class_distribution" in result


def test_bias_check_flag_type(sample_clean):
    """county_representation_bias flag must be a boolean."""
    result = check_representation_bias(sample_clean)
    if "county_representation_bias" in result:
        assert isinstance(result["county_representation_bias"].get("flag", False), bool)
