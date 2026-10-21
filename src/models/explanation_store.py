"""
Explanation store: an append-only audit record of every prediction the API makes.

Each row keeps what a reviewer needs to reconstruct a decision later:
the model version and threshold, the inputs, the probabilities, the visit flag
and the top three reasons (SHAP). The caller's own water point reference is kept
only as a one-way hash, because WPdx IDs encode location.

Rows are never updated or deleted by the application. Retention and deletion
follow docs/ethical_ai_framework.md (section on decommissioning).

Default location: data/explanation_store.duckdb (override with FUNDIFIX_STORE).
"""
import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

STORE_PATH = Path(os.environ.get("FUNDIFIX_STORE", "data/explanation_store.duckdb"))
_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id     VARCHAR PRIMARY KEY,
    logged_at         TIMESTAMP,
    model_name        VARCHAR,
    model_version     INTEGER,
    nf_threshold      DOUBLE,
    point_ref_hash    VARCHAR,
    features          JSON,
    p_functional      DOUBLE,
    p_needs_repair    DOUBLE,
    p_non_functional  DOUBLE,
    predicted_status  VARCHAR,
    flag_for_visit    BOOLEAN,
    top_reasons       JSON
)
"""


def _connect(path: Path = None):
    path = Path(path or STORE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    return con


def hash_ref(ref: Optional[str]) -> Optional[str]:
    """One-way hash of the caller's reference, so it can be matched later but not read."""
    if not ref:
        return None
    return hashlib.sha256(ref.encode("utf-8")).hexdigest()[:16]


def log_predictions(records: list, path: Path = None) -> list:
    """Append prediction records. Each record is a dict with the keys of the table
    except prediction_id and logged_at, which are added here. Returns the new IDs."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows, ids = [], []
    for r in records:
        pid = str(uuid.uuid4())
        ids.append(pid)
        rows.append((pid, now, r["model_name"], int(r["model_version"]), float(r["nf_threshold"]),
                     hash_ref(r.get("point_ref")), json.dumps(r["features"], default=str),
                     r["p_functional"], r["p_needs_repair"], r["p_non_functional"],
                     r["predicted_status"], bool(r["flag_for_visit"]), json.dumps(r["top_reasons"])))
    with _LOCK:
        con = _connect(path)
        try:
            con.executemany("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        finally:
            con.close()
    return ids


def get_prediction(prediction_id: str, path: Path = None) -> Optional[dict]:
    with _LOCK:
        con = _connect(path)
        try:
            df = con.execute("SELECT * FROM predictions WHERE prediction_id = ?", [prediction_id]).df()
        finally:
            con.close()
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    row["logged_at"] = str(row["logged_at"])
    row["features"] = json.loads(row["features"])
    row["top_reasons"] = json.loads(row["top_reasons"])
    return row


def load_features(path: Path = None) -> pd.DataFrame:
    """All logged inputs and outputs as a flat table. Used by the monitoring job."""
    path = Path(path or STORE_PATH)
    if not path.exists():
        return pd.DataFrame()
    con = duckdb.connect(str(path), read_only=True)
    try:
        df = con.execute("SELECT * FROM predictions ORDER BY logged_at").df()
    finally:
        con.close()
    if df.empty:
        return df
    feats = pd.DataFrame([json.loads(f) for f in df["features"]])
    return pd.concat([df.drop(columns=["features"]).reset_index(drop=True), feats], axis=1)


def summary(path: Path = None) -> dict:
    df = load_features(path)
    if df.empty:
        return {"rows": 0}
    return {"rows": int(len(df)), "first": str(df["logged_at"].min()), "last": str(df["logged_at"].max()),
            "model_versions": sorted(int(v) for v in df["model_version"].unique()),
            "share_flagged": round(float(df["flag_for_visit"].mean()), 4)}
