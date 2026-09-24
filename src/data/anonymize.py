"""
Anonymization: remove personal data before export.
Privacy Plan from Module 2, Section 6.1.

What counts as personal data here (Kenya Data Protection Act, 2019, s.2:
information about an identified or identifiable natural person):
  - notes: free-text water point names. Some are named after individuals,
    e.g. "Mama Naomi Spring". The cleaning step already drops this field.
    It is removed here again in case the export is run on raw data.

Removed as a precaution, not a legal requirement:
  - lat_deg, lon_deg: exact GPS points are not needed outside the project.

"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

PERSONAL_NAME_FIELDS = ["notes"]
PRECAUTION_FIELDS = ["lat_deg", "lon_deg"]
DROP_FOR_EXPORT = PERSONAL_NAME_FIELDS + PRECAUTION_FIELDS


def anonymize_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Return an export-safe copy with personal-name and GPS fields removed."""
    present = [c for c in DROP_FOR_EXPORT if c in df.columns]
    export = df.drop(columns=present)
    for col in present:
        logger.info(f"Dropped {col} for export")
    return export
