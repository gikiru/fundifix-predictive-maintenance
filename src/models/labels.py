"""
Readable names for model inputs. Kept free of heavy imports so the API can use it.
"""
from src.models.config import CATEGORICAL_FEATURES

LABELS = {
    "age_at_report": "Age at report (years)",
    "install_year_missing": "Install year not recorded",
    "local_population": "Population within 1 km",
    "assigned_population": "Population assigned to point",
    "usage_cap": "Usage capacity (people)",
    "criticality": "Criticality score",
    "pressure": "Demand pressure",
    "distance_to_primary": "Distance to primary road",
    "distance_to_secondary": "Distance to secondary road",
    "distance_to_tertiary": "Distance to tertiary road",
    "distance_to_city": "Distance to city",
    "distance_to_town": "Distance to town",
    "water_source_clean": "Water source type",
    "water_tech_clean": "Pump / technology type",
    "management_clean": "Management type",
    "is_urban": "Urban or rural",
}


def original_feature(name: str) -> str:
    """Map a one-hot column (e.g. water_tech_clean_Hand Pump) back to its source feature."""
    for c in CATEGORICAL_FEATURES:
        if name.startswith(c + "_"):
            return c
    return name


def readable(name: str) -> str:
    base = original_feature(name)
    if base == name:
        return LABELS.get(name, name)
    return f"{LABELS[base]}: {name[len(base) + 1:]}"
