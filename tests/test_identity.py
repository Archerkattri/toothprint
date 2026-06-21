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


# --- 3D mesh identification (open3d) ---------------------------------------

def _subject(seed=0, n=2000):
    """A distinct synthetic arch (per-subject ellipsoid anisotropy + surface bumps)
    with stable principal axes — the regime where rigid identification applies."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    radii = np.array([18.0, 12.0, 7.0]) + rng.uniform(-3, 3, 3)
    bump = 1 + 0.22 * np.sin(rng.uniform(1, 5) * v[:, 0]) * np.cos(rng.uniform(1, 5) * v[:, 1])
    return v * radii * bump[:, None] + rng.normal(0, 0.05, (n, 3))


def _reproject(pts, seed):
    rng = np.random.default_rng(seed)
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax); a = 0.25
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)
    return pts @ R.T + rng.uniform(-4, 4, 3) + rng.normal(0, 0.06, pts.shape)


def test_align_rigid_genuine_low_impostor_high_after_bestfit():
    pytest.importorskip("open3d")
    from toothprint.identity.mesh import align_rigid
    # genuine re-scan (reposed + noised copy of subject 2) vs an impostor (subject 5)
    query = _reproject(_subject(seed=2), seed=9)
    _, d_gen = align_rigid(query, _subject(seed=2), voxel_size=1.0)
    _, d_imp = align_rigid(query, _subject(seed=5), voxel_size=1.0)
    # even given its BEST rigid alignment, the impostor stays well above the genuine
    assert d_gen < 0.5 and d_imp > 2.0 * d_gen


def test_align_rigid_bad_shape_raises():
    pytest.importorskip("open3d")
    from toothprint.identity.mesh import align_rigid
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        align_rigid(np.zeros((10, 2)), _subject(0))


def test_identify_surface_genuine_is_best():
    pytest.importorskip("open3d")
    from toothprint.identity.mesh import identify_surface
    gallery = [_subject(seed=s) for s in (1, 2, 3)]
    query = _reproject(_subject(seed=2), seed=4)
    row = identify_surface(query, gallery, voxel_size=1.0)
    assert int(np.argmin(row)) == 1     # subject 2 (gallery index 1) is the match


def test_score_to_surface_genuine_sub_noise_impostor_high():
    pytest.importorskip("open3d")
    trimesh = pytest.importorskip("trimesh")
    from toothprint.identity.mesh import score_to_surface
    rng = np.random.default_rng(0)
    gen = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    gen.vertices = gen.vertices * np.array([1.0, 0.7, 0.5])      # anisotropic -> stable axes
    imp = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    imp.vertices = imp.vertices * np.array([0.6, 1.0, 0.85])     # a different shape
    qbase = np.asarray(gen.sample(3000))                          # query = surface re-sample
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax); a = 0.2
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)
    q = qbase @ R.T + rng.uniform(-4, 4, 3) + rng.normal(0, 0.05, qbase.shape)
    d_gen = score_to_surface(q, np.asarray(gen.vertices), np.asarray(gen.faces), voxel_size=1.0)
    d_imp = score_to_surface(q, np.asarray(imp.vertices), np.asarray(imp.faces), voxel_size=1.0)
    # point-to-surface removes the sampling floor: genuine ~scan-noise, impostor far above
    assert d_gen < 0.3 and d_imp > 3.0 * d_gen
