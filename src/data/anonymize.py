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
  - activity_id: some values are enumerator codes (e.g. "Nyamira-Enumerator-962UET").

Replaced (Final Project, reconciliation item C5):
  - wpdx_id: WPdx IDs are Open Location Codes (plus codes) that locate a water
    point to a few metres, so they act as GPS. They are replaced with neutral IDs
    (WPX-00001, ...). The mapping is not exported.

Kept on purpose:
  - clean_adm1/2/3 (county, sub-county, ward): public place names that
    describe thousands of people, not individuals. Needed for analysis.
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

PERSONAL_NAME_FIELDS = ["notes"]
PRECAUTION_FIELDS = ["lat_deg", "lon_deg", "activity_id"]
LOCATION_CODED_ID = "wpdx_id"
DROP_FOR_EXPORT = PERSONAL_NAME_FIELDS + PRECAUTION_FIELDS


def anonymize_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Return an export-safe copy with personal-name, GPS and location-coded fields removed."""
    present = [c for c in DROP_FOR_EXPORT if c in df.columns]
    export = df.drop(columns=present)
    for col in present:
        logger.info(f"Dropped {col} for export")
    if LOCATION_CODED_ID in export.columns:
        export[LOCATION_CODED_ID] = [f"WPX-{i:05d}" for i in range(1, len(export) + 1)]
        logger.info(f"Replaced {LOCATION_CODED_ID} with neutral IDs for export")
    return export
