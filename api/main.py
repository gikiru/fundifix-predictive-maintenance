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
from src.models.explain import original_feature, readable

NF = CLASSES.index("Non-Functional")
STATE = {}


class WaterPoint(BaseModel):
    """One water point. Distances are in metres, as in WPdx. Unknown values can be left out."""
    water_point_id: Optional[str] = Field(None, examples=["6G9VWV9J+J84"])
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
    return PredictResponse(model_version=STATE["meta"]["version"], nf_threshold=thr, predictions=preds)
