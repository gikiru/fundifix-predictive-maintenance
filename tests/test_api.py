"""
Tests for the /predict API. Needs a trained model: run python -m src.models.train first.
Run from repo root: pytest tests/test_api.py -v
"""
from pathlib import Path

import pytest

MODEL = Path("models/fundifix_model.joblib")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="train the model first")

EXAMPLE = {"water_point_id": "test-1", "install_year": 2008, "local_population": 850,
           "usage_cap": 500, "distance_to_primary": 12000, "distance_to_secondary": 4000,
           "distance_to_tertiary": 900, "distance_to_town": 8000,
           "water_source": "Borehole/Tubewell", "water_tech": "Hand Pump - Afridev",
           "management": "Community Management", "is_urban": False}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_valid_probabilities(client):
    r = client.post("/predict", json={"water_points": [EXAMPLE]})
    assert r.status_code == 200
    p = r.json()["predictions"][0]
    assert p["predicted_status"] in ["Functional", "Needs Repair", "Non-Functional"]
    assert abs(sum(p["probabilities"].values()) - 1) < 1e-3
    assert len(p["top_reasons"]) == 3


def test_flag_matches_threshold(client):
    body = client.post("/predict", json={"water_points": [EXAMPLE]}).json()
    p = body["predictions"][0]
    assert p["flag_for_visit"] == (p["p_non_functional"] >= body["nf_threshold"])


def test_missing_optional_fields_are_allowed(client):
    minimal = {k: EXAMPLE[k] for k in ["distance_to_primary", "distance_to_secondary",
                                       "distance_to_tertiary", "distance_to_town"]}
    assert client.post("/predict", json={"water_points": [minimal]}).status_code == 200


def test_unknown_category_is_handled(client):
    odd = {**EXAMPLE, "water_tech": "Solar Pump (new type)"}
    assert client.post("/predict", json={"water_points": [odd]}).status_code == 200


def test_invalid_input_rejected(client):
    bad = {**EXAMPLE, "distance_to_town": -5}
    assert client.post("/predict", json={"water_points": [bad]}).status_code == 422
