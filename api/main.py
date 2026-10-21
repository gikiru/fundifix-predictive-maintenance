"""
FundiFix water point status API.

Run from repo root:   uvicorn api.main:app --reload
Try it in a browser:  http://127.0.0.1:8000/docs
"""
import json
from contextlib import asynccontextmanager
from datetime import date
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.config import CLASSES, FEATURES, MODEL_DIR, MODEL_PATH
from src.models.data import add_derived_features
from src.models.labels import original_feature, readable
from src.models import explanation_store as store

NF = CLASSES.index("Non-Functional")
STATE = {}


class WaterPoint(BaseModel):
    """One water point. Distances are in metres, as in WPdx. Unknown values can be left out."""
    water_point_id: Optional[str] = Field(None, examples=["WP-0001"],
                                           description="Your own reference. Stored only as a one-way hash.")
    install_year: Optional[int] = Field(None, ge=1900, le=2100, examples=[2008])
    assessment_year: Optional[int] = Field(None, ge=1900, le=2100,
                                           description="Year the status is estimated for. Defaults to this year.")
    local_population: Optional[float] = Field(None, ge=0, examples=[850])
    assigned_population: Optional[float] = Field(None, ge=0, examples=[300])
    usage_cap: Optional[float] = Field(None, ge=0, examples=[500])
    criticality: Optional[float] = Field(None, ge=0, examples=[0.4])
    pressure: Optional[float] = Field(None, ge=0, examples=[0.6])
    distance_to_primary: float = Field(..., ge=0, examples=[12000])
    distance_to_secondary: float = Field(..., ge=0, examples=[4000])
    distance_to_tertiary: float = Field(..., ge=0, examples=[900])
    distance_to_city: Optional[float] = Field(None, ge=0, examples=[45000])
    distance_to_town: float = Field(..., ge=0, examples=[8000])
    water_source: Optional[str] = Field(None, examples=["Borehole/Tubewell"])
    water_tech: Optional[str] = Field(None, examples=["Hand Pump - Afridev"])
    management: Optional[str] = Field(None, examples=["Community Management"])
    is_urban: bool = Field(False)


class PredictRequest(BaseModel):
    water_points: List[WaterPoint] = Field(..., min_length=1, max_length=1000)


class Reason(BaseModel):
    feature: str
    effect_on_non_functional: float


class Prediction(BaseModel):
    prediction_id: Optional[str] = Field(None, description="Key for GET /explanations/{prediction_id}")
    water_point_id: Optional[str]
    predicted_status: str
    probabilities: dict
    p_non_functional: float
    flag_for_visit: bool
    top_reasons: List[Reason]


class PredictResponse(BaseModel):
    model_version: int
    nf_threshold: float
    predictions: List[Prediction]


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["model"] = joblib.load(MODEL_PATH)
    STATE["meta"] = json.loads((MODEL_DIR / "model_meta.json").read_text())
    STATE["explainer"] = shap.TreeExplainer(STATE["model"].named_steps["clf"])
    yield
    STATE.clear()


app = FastAPI(title="FundiFix water point status API", version="1.0", lifespan=lifespan)


def to_frame(points: List[WaterPoint]) -> pd.DataFrame:
    raw = pd.DataFrame([p.model_dump() for p in points]).rename(columns={
        "water_source": "water_source_clean", "water_tech": "water_tech_clean",
        "management": "management_clean"})
    year = raw["assessment_year"].astype("float").fillna(date.today().year)
    raw["install_year"] = raw["install_year"].astype("float")
    return add_derived_features(raw, report_year=year)


def reasons(X: pd.DataFrame, k: int = 3) -> List[List[Reason]]:
    """Top k features pushing each point's Non-Functional probability, from SHAP."""
    prep = STATE["model"].named_steps["prep"]
    X_t = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out())
    vals = STATE["explainer"](X_t).values[:, :, NF]
    out = []
    for i in range(len(X_t)):
        by_feature = pd.Series(vals[i], index=X_t.columns).groupby(
            [original_feature(c) for c in X_t.columns]).sum()
        top = by_feature.reindex(by_feature.abs().sort_values(ascending=False).index)[:k]
        out.append([Reason(feature=readable(f), effect_on_non_functional=round(float(v), 3))
                    for f, v in top.items()])
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.get("/model-info")
def model_info():
    m = STATE["meta"]
    return {k: m[k] for k in ["registered_model", "version", "alias", "family", "classes",
                              "nf_threshold", "training_scope", "test_metrics"]}


@app.get("/explanations/{prediction_id}")
def explanation(prediction_id: str):
    """Audit lookup: the stored inputs, outputs and reasons for one past prediction."""
    row = store.get_prediction(prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No stored prediction with that ID")
    return row


@app.get("/audit/summary")
def audit_summary():
    """How many predictions are stored, over what period, and the share flagged."""
    return store.summary()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        X = to_frame(req.water_points)[FEATURES]
        proba = STATE["model"].predict_proba(X)
        why = reasons(X)
    except Exception as e:                                   # bad input shape or types
        raise HTTPException(status_code=422, detail=str(e))
    thr = STATE["meta"]["nf_threshold"]
    preds = [Prediction(water_point_id=p.water_point_id,
                        predicted_status=CLASSES[int(np.argmax(pr))],
                        probabilities={c: round(float(v), 4) for c, v in zip(CLASSES, pr)},
                        p_non_functional=round(float(pr[NF]), 4),
                        flag_for_visit=bool(pr[NF] >= thr),
                        top_reasons=r)
             for p, pr, r in zip(req.water_points, proba, why)]
    # Audit trail: every prediction and its reasons are stored before the answer is returned.
    # If the store cannot be written, the request fails rather than returning an unlogged decision.
    meta = STATE["meta"]
    records = [{"model_name": meta["registered_model"], "model_version": meta["version"], "nf_threshold": thr,
                "point_ref": p.water_point_id, "features": X.iloc[i].to_dict(),
                "p_functional": pr.probabilities["Functional"], "p_needs_repair": pr.probabilities["Needs Repair"],
                "p_non_functional": pr.p_non_functional, "predicted_status": pr.predicted_status,
                "flag_for_visit": pr.flag_for_visit, "top_reasons": [r.model_dump() for r in pr.top_reasons]}
               for i, (p, pr) in enumerate(zip(req.water_points, preds))]
    try:
        ids = store.log_predictions(records)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Explanation store unavailable, prediction not released: {e}")
    for pr, pid in zip(preds, ids):
        pr.prediction_id = pid
    return PredictResponse(model_version=STATE["meta"]["version"], nf_threshold=thr, predictions=preds)
