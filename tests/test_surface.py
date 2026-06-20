import numpy as np
import pytest

from toothprint.surface.certificate import SurfaceCertificate, certify_surface_change
from toothprint.surface.error import chamfer_distance, icp_align, surface_error
from toothprint.change.conformal import CHANGED, STABLE, UNCERTAIN, ConformalCertifier


def _sphere(n=2000, r=10.0, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * r


# --- error -----------------------------------------------------------------

def test_chamfer_self_zero():
    p = _sphere(500)
    assert chamfer_distance(p, p) == 0.0


def test_chamfer_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        chamfer_distance(np.zeros((0, 3)), _sphere(10))


def test_icp_recovers_translation():
    tgt = _sphere(800, seed=1)
    src = tgt + np.array([2.0, -1.0, 0.5])
    aligned, n = icp_align(src, tgt)
    assert chamfer_distance(aligned, tgt) < 0.5 and n >= 1


def test_icp_estimate_scale_beats_rigid():
    tgt = _sphere(800, seed=2)
    src = tgt * 0.85
    scaled, _ = icp_align(src, tgt, estimate_scale=True, max_iterations=80)
    rigid, _ = icp_align(src, tgt, estimate_scale=False, max_iterations=80)
    assert chamfer_distance(scaled, tgt) < chamfer_distance(rigid, tgt)


def test_surface_error_fields_and_denoise():
    ref = _sphere(3000, seed=3)
    noisy = _sphere(2000, seed=4) + np.random.default_rng(5).normal(0, 0.3, (2000, 3))
    e = surface_error(noisy, ref, run_icp=True)
    assert e.chamfer_mm > 0 and e.hausdorff_mm >= e.chamfer_mm and e.rms_mm > 0
    assert e.n_source == 2000 and e.icp_iterations >= 1


def test_surface_error_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        surface_error(np.zeros((0, 3)), _sphere(10))


def test_surface_error_nonfinite_raises():
    bad = _sphere(10).copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        surface_error(bad, _sphere(10))


def test_surface_error_no_icp_path():
    e = surface_error(_sphere(50), _sphere(50, seed=9), run_icp=False)
    assert e.icp_iterations == 0


# --- certificate -----------------------------------------------------------

def test_certify_surface_labels():
    tight = ConformalCertifier(q_lo=0.1, q_hi=0.1, alpha=0.1)
    assert certify_surface_change(0.05, tight).label == STABLE
    assert certify_surface_change(1.0, tight).label == CHANGED
    assert certify_surface_change(0.6, tight).label == UNCERTAIN


def test_certify_surface_returns_dataclass():
    cert = ConformalCertifier(q_lo=0.1, q_hi=0.1, alpha=0.1)
    out = certify_surface_change(1.2, cert)
    assert isinstance(out, SurfaceCertificate)
    assert out.interval_mm[0] >= 0.0


def test_certify_surface_bad_thresholds():
    cert = ConformalCertifier(q_lo=0.1, q_hi=0.1, alpha=0.1)
    with pytest.raises(ValueError, match="must be <"):
        certify_surface_change(0.5, cert, stable_threshold_mm=0.9, change_threshold_mm=0.5)


# --- meshing (open3d) ------------------------------------------------------

def test_poisson_refine_denoises_sphere():
    o3d = pytest.importorskip("open3d")
    from toothprint.surface.meshing import poisson_refine
    clean = _sphere(6000, seed=6)
    noisy = clean + np.random.default_rng(7).normal(0, 0.3, clean.shape)
    refined = poisson_refine(noisy, depth=8)
    raw_err = float(np.abs(np.linalg.norm(noisy, axis=1) - 10.0).mean())
    ref_err = float(np.abs(np.linalg.norm(refined, axis=1) - 10.0).mean())
    assert ref_err < raw_err


def test_poisson_refine_validation_raises():
    from toothprint.surface.meshing import poisson_refine
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        poisson_refine(np.zeros((50, 2)))
    with pytest.raises(ValueError, match="at least 30"):
        poisson_refine(np.zeros((5, 3)))
    pytest.importorskip("open3d")
    with pytest.raises(ValueError, match="positive"):
        poisson_refine(_sphere(50), depth=0)


def test_poisson_refine_open3d_missing_raises(monkeypatch):
    import sys
    from toothprint.surface.meshing import poisson_refine
    monkeypatch.setitem(sys.modules, "open3d", None)
    with pytest.raises(RuntimeError, match="open3d is required"):
        poisson_refine(_sphere(50))


def test_poisson_refine_empty_mesh_raises(monkeypatch):
    o3d = pytest.importorskip("open3d")
    from toothprint.surface.meshing import poisson_refine
    empty = (o3d.geometry.TriangleMesh(), o3d.utility.DoubleVector([]))
    monkeypatch.setattr(o3d.geometry.TriangleMesh, "create_from_point_cloud_poisson",
                        staticmethod(lambda *a, **k: empty))
    with pytest.raises(RuntimeError, match="empty mesh"):
        poisson_refine(_sphere(200))
