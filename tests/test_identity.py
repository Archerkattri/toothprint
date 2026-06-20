import numpy as np
import pytest

from toothprint.identity.constellation import (
    _normalize, constellation, icp_residual, identify as identify_radiograph,
)
from toothprint.identity.metrics import identification_metrics, rank1_match


# --- metrics ---------------------------------------------------------------

def test_rank1_match():
    assert rank1_match([0.5, 0.1, 0.9]) == 1


def test_rank1_match_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        rank1_match([])


def test_identification_metrics_perfect():
    s = np.array([[0.1, 0.8, 0.9], [0.7, 0.1, 0.8], [0.9, 0.7, 0.1]])
    labels = ["a", "b", "c"]
    m = identification_metrics(s, labels, labels)
    assert m["rank1_accuracy"] == 1.0
    assert m["genuine_max"] < m["impostor_min"]
    assert m["decidability"] > 2.0


def test_identification_metrics_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        identification_metrics(np.zeros((2, 3)), ["a", "b"], ["a", "b"])


def test_identification_metrics_no_genuine_nan():
    m = identification_metrics(np.array([[0.3, 0.4]]), ["q"], ["a", "b"])
    assert np.isnan(m["genuine_mean"]) and m["rank1_accuracy"] == 0.0


def test_identification_metrics_zero_variance_dprime_zero():
    m = identification_metrics(np.array([[0.5, 0.5]]), ["a"], ["a", "b"])
    assert m["decidability"] == 0.0


# --- 2D constellation ------------------------------------------------------

def _ann(seed=0, n=6):
    rng = np.random.default_rng(seed)
    teeth = []
    for k in range(n):
        cx = 50 + k * 40
        teeth.append({"tooth_id": str(k),
                      "cej": [[cx, 100.0 + rng.normal(0, 5)], [cx + 15, 100.0]],
                      "crest_line": [[cx, 130.0 + rng.normal(0, 5)], [cx + 15, 130.0]],
                      "apex": [[cx + 7, 200.0 + rng.normal(0, 5)]]})
    return {"teeth": teeth}


def test_constellation_collects():
    assert constellation(_ann(0, n=3)).shape == (15, 2)


def test_constellation_empty_raises():
    with pytest.raises(ValueError, match="no landmarks"):
        constellation({"teeth": [{"tooth_id": "1", "cej": [], "crest_line": []}]})


def test_normalize_unit_rms():
    c = np.array([[0.0, 0.0], [3.0, 4.0]])
    normed, rms = _normalize(c)
    assert rms > 0 and np.allclose(normed.mean(0), 0, atol=1e-9)


def test_normalize_degenerate_single_point():
    normed, rms = _normalize(np.array([[5.0, 5.0]]))
    assert rms == 1.0


def test_icp_genuine_low_impostor_high():
    base = constellation(_ann(1))
    rng = np.random.default_rng(9)
    ang = 0.1
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    genuine = 1.1 * base @ R.T + np.array([10.0, -5.0]) + rng.normal(0, 1.0, base.shape)
    impostor = constellation(_ann(2))
    assert icp_residual(genuine, base) < icp_residual(impostor, base)


def test_icp_identical_zero():
    c = constellation(_ann(4))
    assert icp_residual(c, c) < 1e-6


def test_identify_radiograph_rank1():
    cons = [constellation(_ann(s)) for s in (5, 6, 7)]
    rng = np.random.default_rng(0)
    q = 1.05 * cons[1] + np.array([5.0, 5.0]) + rng.normal(0, 0.5, cons[1].shape)
    assert int(np.argmin(identify_radiograph(q, cons))) == 1


# --- 3D mesh (open3d) ------------------------------------------------------

def _arch(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    t = np.linspace(-1.2, 1.2, n)
    pts = np.stack([t * 18.0, (t ** 2) * 12.0, np.sin(t * 6.0) * 2.0 + rng.normal(0, 0.05, n)], axis=1)
    for _ in range(3):
        c = rng.uniform(-18, 18, 3) * np.array([1, 0.6, 0.1])
        pts[:, 2] += 2.0 * np.exp(-(np.linalg.norm(pts - c, axis=1) ** 2) / 8.0)
    return pts


def test_to_point_cloud_bad_shape():
    pytest.importorskip("open3d")
    from toothprint.identity.mesh import to_point_cloud
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        to_point_cloud(np.zeros((10, 2)), 1.0)


def test_fpfh_33_dims():
    pytest.importorskip("open3d")
    from toothprint.identity.mesh import compute_fpfh, to_point_cloud
    f = compute_fpfh(to_point_cloud(_arch(0), 1.0), 1.0)
    assert np.asarray(f.data).shape[0] == 33


def test_identify_scan_genuine_is_best():
    pytest.importorskip("open3d")
    from toothprint.identity.mesh import enroll, identify as identify_scan
    voxel = 1.0
    rng = np.random.default_rng(1)
    gallery = [enroll(_arch(seed=s), voxel) for s in (1, 2, 3)]
    ang = 0.1
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    query = _arch(seed=2) @ R.T + rng.uniform(-3, 3, 3) + rng.normal(0, 0.05, (1500, 3))
    row = identify_scan(query, gallery, voxel)
    assert int(np.argmin(row)) == 1  # subject 2 is gallery index 1
