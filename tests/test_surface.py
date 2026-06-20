import numpy as np
import pytest

from toothprint.surface.certificate import SurfaceCertificate, certify_surface_change
from toothprint.surface.error import (
    assign_regions, chamfer_distance, icp_align, noise_floor_sq,
    regional_displacements, surface_displacement, surface_error,
)
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


# --- differential displacement (de-biased) ---------------------------------

def _displace(pts, mm):
    c = pts.mean(0)
    u = (pts - c) / np.clip(np.linalg.norm(pts - c, axis=1, keepdims=True), 1e-9, None)
    return pts + u * mm


def test_noise_floor_sq_estimates_power():
    rng = np.random.default_rng(0)
    base = _sphere(2000)
    sigma = 0.2
    pairs = [(base + rng.normal(0, sigma, base.shape), base + rng.normal(0, sigma, base.shape))
             for _ in range(8)]
    f = noise_floor_sq(pairs)
    # E[mean ||n1 - n0||^2] = 3 axes * 2 * sigma^2 = 6 sigma^2
    assert abs(f - 6 * sigma ** 2) < 0.02


def test_noise_floor_sq_empty_raises():
    with pytest.raises(ValueError, match="at least one stable pair"):
        noise_floor_sq([])


def test_noise_floor_sq_bad_pair_raises():
    with pytest.raises(ValueError, match="corresponded clouds"):
        noise_floor_sq([(_sphere(10), _sphere(11))])


def test_surface_displacement_debiases_noise_to_zero():
    # The raw mean-norm of a stable (no-change) noisy pair is a large positive
    # bias (~2.3 sigma); de-biasing with the estimated floor drives it to ~0.
    rng = np.random.default_rng(1)
    base = _sphere(2000)
    sigma = 0.2
    floor = noise_floor_sq([(base + rng.normal(0, sigma, base.shape),
                             base + rng.normal(0, sigma, base.shape)) for _ in range(12)])
    a = base + rng.normal(0, sigma, base.shape)
    b = base + rng.normal(0, sigma, base.shape)  # no real change
    raw = float(np.linalg.norm(b - a, axis=1).mean())
    debiased = surface_displacement(a, b, noise_floor_sq=floor)
    assert raw > 0.4                  # rectified-noise floor is large
    assert debiased < 0.1             # de-biasing removes it


def test_surface_displacement_recovers_change_under_noise():
    # A 1mm change at 0.2mm noise (where the raw certificate collapses) is
    # recovered to ~1mm by the de-biased estimator.
    rng = np.random.default_rng(2)
    base = _sphere(2000)
    sigma = 0.2
    floor = noise_floor_sq([(base + rng.normal(0, sigma, base.shape),
                             base + rng.normal(0, sigma, base.shape)) for _ in range(12)])
    t0 = base + rng.normal(0, sigma, base.shape)
    t1 = _displace(base, 1.0) + rng.normal(0, sigma, base.shape)
    assert abs(surface_displacement(t0, t1, noise_floor_sq=floor) - 1.0) < 0.1


def test_surface_displacement_shape_mismatch_raises():
    with pytest.raises(ValueError, match="equal shape"):
        surface_displacement(_sphere(10), _sphere(11))


def test_surface_displacement_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        surface_displacement(np.zeros((0, 3)), np.zeros((0, 3)))


def test_surface_displacement_nonfinite_raises():
    a = _sphere(10); b = _sphere(10).copy(); b[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        surface_displacement(a, b)


# --- regional (localized-change) detection ---------------------------------

def test_assign_regions_partitions_deterministically():
    pts = _sphere(2000)
    lab = assign_regions(pts, n_regions=12, seed=0)
    assert lab.shape == (2000,) and lab.min() >= 0 and lab.max() < 12
    assert len(np.unique(lab)) == 12              # FPS spreads centres; all non-empty
    assert np.array_equal(lab, assign_regions(pts, n_regions=12, seed=0))


def test_assign_regions_validation_and_clamp():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        assign_regions(np.zeros((10, 2)))
    with pytest.raises(ValueError, match=">= 1"):
        assign_regions(_sphere(10), n_regions=0)
    lab = assign_regions(_sphere(5), n_regions=50)  # clamp k to N
    assert lab.max() < 5


def test_regional_localizes_change_that_global_dilutes():
    rng = np.random.default_rng(0)
    base = _sphere(2000)
    sigma, K = 0.1, 12
    lab = assign_regions(base, n_regions=K, seed=0)
    # a localized 1mm change: push one region's patch radially outward
    patch = lab == 0
    t1 = base.copy()
    t1[patch] = base[patch] + base[patch] / np.linalg.norm(base[patch], axis=1, keepdims=True) * 1.0
    stable = [(base + rng.normal(0, sigma, base.shape), base + rng.normal(0, sigma, base.shape))
              for _ in range(8)]
    floors = [noise_floor_sq([(a[lab == r], b[lab == r]) for a, b in stable]) for r in range(K)]
    a = base + rng.normal(0, sigma, base.shape)
    b = t1 + rng.normal(0, sigma, base.shape)
    d = regional_displacements(a, b, lab, floors)
    assert d.argmax() == 0 and d[0] > 0.8         # localizes + recovers ~full 1mm
    g = surface_displacement(a, b, noise_floor_sq=noise_floor_sq(stable))
    assert g < 0.5 and d.max() > 2 * g            # global dilutes; regional recovers it


def test_regional_displacements_validation():
    base = _sphere(50); lab = assign_regions(base, 4)
    with pytest.raises(ValueError, match="equal shape"):
        regional_displacements(base, _sphere(51), lab, [0, 0, 0, 0])
    with pytest.raises(ValueError, match="one entry per point"):
        regional_displacements(base, base.copy(), lab[:10], [0, 0, 0, 0])


def test_regional_empty_region_reads_zero():
    base = _sphere(50)
    lab = np.zeros(50, int)                        # region 1 has no points
    d = regional_displacements(base, base.copy(), lab, [0.0, 0.0])
    assert d[1] == 0.0


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
