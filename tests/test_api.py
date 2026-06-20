import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_identify_radiograph_picks_genuine():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 30, (20, 2)).tolist()
    other = rng.normal(0, 30, (20, 2)).tolist()
    # query is base under a similarity transform + small jitter
    b = np.asarray(base)
    ang = 0.1
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    query = (1.1 * b @ R.T + np.array([8.0, -4.0]) + rng.normal(0, 0.5, b.shape)).tolist()
    r = client.post("/api/identify/radiograph", json={
        "query": query,
        "gallery": [{"label": "subject_A", "points": base},
                    {"label": "subject_B", "points": other}],
    })
    assert r.status_code == 200
    assert r.json()["match"] == "subject_A"
    assert len(r.json()["ranking"]) == 2


def test_certify_change_labels():
    r = client.post("/api/certify/change", json={
        "measured_px": 30.0, "q_lo": 0.5, "q_hi": 0.5, "tau": 6.0})
    assert r.json()["label"] == "changed"
    r2 = client.post("/api/certify/change", json={
        "measured_px": 0.1, "q_lo": 0.5, "q_hi": 0.5, "tau": 6.0})
    assert r2.json()["label"] == "stable"


def test_certify_surface_labels():
    r = client.post("/api/certify/surface", json={
        "measured_mm": 1.2, "q_lo": 0.1, "q_hi": 0.1})
    assert r.json()["label"] == "changed"
    r2 = client.post("/api/certify/surface", json={
        "measured_mm": 0.05, "q_lo": 0.1, "q_hi": 0.1})
    assert r2.json()["label"] == "stable"


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and b"ToothPrint" in r.content
