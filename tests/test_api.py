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


# --- hardening: bounded + finite input -------------------------------------

def test_formats_endpoint():
    r = client.get("/api/formats")
    assert r.status_code == 200 and "scan" in r.json()["supported"]


def test_security_headers_present():
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"


def test_reject_nonfinite_points():
    # raw body: 1e309 is valid JSON but parses to +inf -> must be rejected cleanly (422, not 500)
    body = ('{"query": [[1e309, 0.0], [1.0, 2.0]], '
            '"gallery": [{"label": "a", "points": [[0.0, 0.0], [1.0, 1.0]]}]}')
    r = client.post("/api/identify/radiograph", content=body,
                    headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_reject_wrong_point_dim():
    r = client.post("/api/identify/radiograph", json={
        "query": [[0.0, 1.0, 2.0]],
        "gallery": [{"label": "a", "points": [[0.0, 0.0]]}]})
    assert r.status_code == 422


def test_reject_empty_gallery():
    r = client.post("/api/identify/radiograph", json={
        "query": [[0.0, 0.0]], "gallery": []})
    assert r.status_code == 422


def test_reject_alpha_out_of_range():
    for a in (0.0, 1.0, -0.1, 1.5):
        r = client.post("/api/certify/change", json={
            "measured_px": 5.0, "q_lo": 0.5, "q_hi": 0.5, "alpha": a})
        assert r.status_code == 422


def test_reject_inverted_quantiles():
    r = client.post("/api/certify/surface", json={
        "measured_mm": 1.0, "q_lo": 0.9, "q_hi": 0.1})
    assert r.status_code == 422
    r2 = client.post("/api/certify/change", json={
        "measured_px": 5.0, "q_lo": 0.9, "q_hi": 0.1})
    assert r2.status_code == 422


def test_inspect_volume(tmp_path):
    pytest.importorskip("nibabel")
    import nibabel as nib
    vol = np.random.default_rng(0).random((8, 9, 7)).astype(np.float32)
    p = tmp_path / "v.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.diag([0.3, 0.3, 0.3, 1])), str(p))
    with open(p, "rb") as fh:
        r = client.post("/api/inspect", files={"file": ("v.nii.gz", fh, "application/octet-stream")})
    assert r.status_code == 200 and r.json()["kind"] == "volume"
    assert r.json()["shape"] == [8, 9, 7]


def test_oversize_request_rejected():
    huge = [[0.0, 0.0]] * 6000          # > MAX_POINTS
    r = client.post("/api/identify/radiograph", json={
        "query": huge, "gallery": [{"label": "a", "points": [[0.0, 0.0]]}]})
    assert r.status_code == 422


# --- safe file ingest ------------------------------------------------------

def test_inspect_scan(tmp_path):
    pytest.importorskip("trimesh")
    import trimesh
    p = tmp_path / "m.stl"
    trimesh.creation.box(extents=[10, 12, 8]).export(str(p))
    with open(p, "rb") as fh:
        r = client.post("/api/inspect", files={"file": ("m.stl", fh, "application/octet-stream")})
    assert r.status_code == 200 and r.json()["kind"] == "scan"
    assert r.json()["n_faces"] == 12


def test_inspect_radiograph(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    p = tmp_path / "x.png"
    Image.fromarray((np.random.default_rng(0).random((40, 48)) * 255).astype(np.uint8)).save(str(p))
    with open(p, "rb") as fh:
        r = client.post("/api/inspect", files={"file": ("x.png", fh, "image/png")})
    assert r.status_code == 200 and r.json()["kind"] == "radiograph"
    assert r.json()["shape"] == [40, 48]


def test_inspect_rejects_garbage(tmp_path):
    p = tmp_path / "j.xyz"; p.write_bytes(b"\x00\x01 not a medical file")
    with open(p, "rb") as fh:
        r = client.post("/api/inspect", files={"file": ("j.xyz", fh, "application/octet-stream")})
    assert r.status_code == 422


def test_request_body_size_cap(monkeypatch):
    monkeypatch.setattr("api.main.MAX_REQUEST_BYTES", 5)
    r = client.post("/api/certify/change", json={"measured_px": 5.0, "q_lo": 0.5, "q_hi": 0.5})
    assert r.status_code == 413


def test_upload_size_cap(tmp_path, monkeypatch):
    pytest.importorskip("trimesh")
    import trimesh
    monkeypatch.setattr("api.main.MAX_UPLOAD_BYTES", 4)
    p = tmp_path / "m.stl"; trimesh.creation.box().export(str(p))
    with open(p, "rb") as fh:
        r = client.post("/api/inspect", files={"file": ("m.stl", fh, "application/octet-stream")})
    assert r.status_code == 413


# --- desktop studio + scan identification ----------------------------------

def test_studio_served():
    r = client.get("/studio")
    assert r.status_code == 200 and b"STUDIO" in r.content


def _mesh_bytes(tmp_path, name, shape):
    import trimesh
    p = tmp_path / name
    shape.export(str(p))
    return ("file_or_files", (name, open(p, "rb"), "application/octet-stream"))


def test_identify_scan_matches_same_arch(tmp_path):
    pytest.importorskip("trimesh")
    pytest.importorskip("open3d")
    import trimesh
    q = tmp_path / "q.stl"; trimesh.creation.icosphere(subdivisions=5, radius=10).export(str(q))  # >4000 verts -> exercises downsample
    g = tmp_path / "g.stl"; trimesh.creation.icosphere(subdivisions=3, radius=10).export(str(g))
    o = tmp_path / "o.stl"; trimesh.creation.box(extents=[20, 8, 6]).export(str(o))
    files = [("files", ("q.stl", open(q, "rb"), "application/octet-stream")),
             ("files", ("g.stl", open(g, "rb"), "application/octet-stream")),
             ("files", ("o.stl", open(o, "rb"), "application/octet-stream"))]
    r = client.post("/api/identify/scan", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["match"] == "g.stl" and body["verdict"] == "same person"
    assert len(body["ranking"]) == 2


def test_identify_scan_needs_two(tmp_path):
    pytest.importorskip("trimesh")
    import trimesh
    q = tmp_path / "q.stl"; trimesh.creation.box().export(str(q))
    r = client.post("/api/identify/scan",
                    files=[("files", ("q.stl", open(q, "rb"), "application/octet-stream"))])
    assert r.status_code == 422


def test_identify_scan_rejects_bad_file(tmp_path):
    pytest.importorskip("trimesh")
    import trimesh
    q = tmp_path / "q.stl"; trimesh.creation.box().export(str(q))
    bad = tmp_path / "bad.stl"; bad.write_bytes(b"not a real mesh")
    r = client.post("/api/identify/scan", files=[
        ("files", ("q.stl", open(q, "rb"), "application/octet-stream")),
        ("files", ("bad.stl", open(bad, "rb"), "application/octet-stream"))])
    assert r.status_code == 422


def test_identify_scan_upload_cap(tmp_path, monkeypatch):
    pytest.importorskip("trimesh")
    import trimesh
    monkeypatch.setattr("api.main.MAX_UPLOAD_BYTES", 4)
    q = tmp_path / "q.stl"; trimesh.creation.box().export(str(q))
    g = tmp_path / "g.stl"; trimesh.creation.box().export(str(g))
    r = client.post("/api/identify/scan", files=[
        ("files", ("q.stl", open(q, "rb"), "application/octet-stream")),
        ("files", ("g.stl", open(g, "rb"), "application/octet-stream"))])
    assert r.status_code == 413
