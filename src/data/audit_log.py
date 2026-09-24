"""
Privacy Audit Log: Log all data access and transformation events.
Module 2 Section 6.1 — Privacy Audit Logging requirement.
"""
import logging
import json
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path("logs/privacy_audit.log")

def get_audit_logger() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("privacy_audit")
    if not logger.handlers:
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    return logger


def log_event(event_type: str, dataset: str, details: dict = None) -> None:
    """Log a data access or transformation event as structured JSON."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,   # e.g. INGEST, CLEAN, TRANSFORM, EXPORT, VALIDATE
        "dataset": dataset,
        "details": details or {},
    }
    get_audit_logger().info(json.dumps(record))


if __name__ == "__main__":
    log_event("INGEST", "wpdx_enhanced.csv", {"rows": 21953, "source": "OCHA HDX"})
    log_event("CLEAN", "raw_wpdx", {"rows_out": 21952, "dropped_cols": 14})
    log_event("EXPORT", "wpdx_anonymised_export.csv",
              {"pii_fields_removed": ["lat_deg", "lon_deg", "clean_adm2"],
               "pseudonymised": ["clean_adm3"]})
    print(f"Audit log written to {LOG_FILE}")
