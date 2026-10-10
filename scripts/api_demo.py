"""
API demo: send three example water points to the Module 4 /predict endpoint and print the answers.

1. In one terminal, start the API from the repo root:   python -m uvicorn api.main:app
2. In a second terminal, run:                           python scripts/api_demo.py

The three examples match the sample predictions in the Module 5 deck (WP-1740, WP-1412, WP-0784).
"""
import json
import sys

import httpx

URL = "http://127.0.0.1:8000"

EXAMPLES = [
    {"water_point_id": "WP-1740 (Makueni)", "install_year": 1971, "assessment_year": 2021,
     "local_population": 204, "assigned_population": 204, "criticality": 1.0,
     "distance_to_primary": 23427, "distance_to_secondary": 11930, "distance_to_tertiary": 14720,
     "distance_to_city": 80426, "distance_to_town": 23105,
     "water_source": "Sand or Sub-surface Dam", "water_tech": "Public Tapstand",
     "management": "Community Management", "is_urban": False},
    {"water_point_id": "WP-1412 (Nyamira)", "install_year": 2000, "assessment_year": 2021,
     "local_population": 3045, "assigned_population": 469, "usage_cap": 300,
     "criticality": 0.154023, "pressure": 1.563333,
     "distance_to_primary": 22559, "distance_to_secondary": 5464, "distance_to_tertiary": 781,
     "distance_to_city": 23096, "distance_to_town": 4086,
     "water_source": "Protected Well", "water_tech": "Public Tapstand",
     "management": "Other", "is_urban": False},
    {"water_point_id": "WP-0784 (Kisumu)", "install_year": 2000, "assessment_year": 2021,
     "usage_cap": 250,
     "distance_to_primary": 2087, "distance_to_secondary": 1150, "distance_to_tertiary": 285,
     "distance_to_city": 2044, "distance_to_town": 10660,
     "water_source": "Piped Water", "water_tech": "Public Tapstand",
     "management": "Private Operator/Delegated Management", "is_urban": True},
]


def main():
    try:
        health = httpx.get(f"{URL}/health", timeout=10).json()
    except httpx.ConnectError:
        sys.exit("The API is not running. Start it first: python -m uvicorn api.main:app")
    print("Health:", health)
    info = httpx.get(f"{URL}/model-info", timeout=10).json()
    print(f"Model: {info['registered_model']} v{info['version']} ({info['family']}), "
          f"visit threshold {info['nf_threshold']}\n")

    r = httpx.post(f"{URL}/predict", json={"water_points": EXAMPLES}, timeout=60)
    r.raise_for_status()
    for p in r.json()["predictions"]:
        print(f"{p['water_point_id']}")
        print(f"  Main guess:          {p['predicted_status']}")
        print(f"  Chances:             " + ", ".join(f"{k} {v:.0%}" for k, v in p["probabilities"].items()))
        print(f"  On the visit list:   {'Yes' if p['flag_for_visit'] else 'No'}")
        print("  Top reasons:         " + "; ".join(
            f"{x['feature']} ({'raises' if x['effect_on_non_functional'] > 0 else 'lowers'} risk)"
            for x in p["top_reasons"]))
        print()

    bad = {"water_points": [{**EXAMPLES[0], "distance_to_town": -5}]}
    r = httpx.post(f"{URL}/predict", json=bad, timeout=10)
    print(f"Invalid input check (negative distance): HTTP {r.status_code} "
          f"{'rejected as expected' if r.status_code == 422 else 'NOT rejected'}")
    print("\nRaw JSON for the first water point:")
    print(json.dumps(r.json() if r.status_code != 422 else
                     httpx.post(f"{URL}/predict", json={"water_points": EXAMPLES[:1]}, timeout=60).json(), indent=2))


if __name__ == "__main__":
    main()
