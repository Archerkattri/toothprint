"""Tests for dentalmapcert.surface_error — 14 tests."""
from __future__ import annotations
import numpy as np
import pytest
from dentalmapcert.surface_error import (
    SurfaceError,
    _closest_point_on_segment,
    _point_to_triangle_distance,
    chamfer_distance,
    icp_align,
    point_to_surface_distance,
    surface_error_mm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotation_z(theta_rad: float) -> np.ndarray:
    """3×3 rotation matrix about the z-axis."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _make_cloud(n: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3))


# ---------------------------------------------------------------------------
# ICP tests
# ---------------------------------------------------------------------------

class TestICPAlign:
    def test_icp_identity(self):
        """Identical source and target → R≈I, t≈0, converged."""
        cloud = _make_cloud(20)
        aligned, R, t, n_iters, converged = icp_align(cloud, cloud)
        assert converged
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(aligned, cloud, atol=1e-6)

    def test_icp_estimate_scale_recovers_known_scale(self):
        """Source = a modestly scaled target → similarity ICP recovers the scale.

        Uses identity orientation and a small scale gap so closest-point
        correspondences are good from the start (cold-start similarity ICP is
        local and cannot recover large scale gaps without initialisation).
        """
        target = _make_cloud(40, seed=3)
        source = target * 0.85  # 0.85x scale, same orientation/centre region
        aligned, R, t, n_iters, converged = icp_align(
            source, target, estimate_scale=True, max_iterations=80)
        err = np.linalg.norm(aligned - target, axis=1).mean()
        # Similarity ICP should beat the rigid-only residual (which cannot fix scale).
        rigid, *_ = icp_align(source, target, estimate_scale=False, max_iterations=80)
        rigid_err = np.linalg.norm(rigid - target, axis=1).mean()
        assert err < rigid_err, f"scale ICP ({err}) should beat rigid ({rigid_err})"

    def test_icp_estimate_scale_handles_degenerate_zero_variance(self):
        """All-identical source points (zero variance) → scale falls back to 1.0."""
        source = np.zeros((10, 3))
        target = _make_cloud(10, seed=7)
        aligned, R, t, n_iters, converged = icp_align(
            source, target, estimate_scale=True, max_iterations=3)
        assert np.isfinite(aligned).all()

    def test_icp_pure_translation(self):
        """Source = target + [1,0,0] → ICP recovers the translation."""
        target = _make_cloud(30, seed=1)
        offset = np.array([1.0, 0.0, 0.0])
        source = target + offset
        aligned, R, t, n_iters, converged = icp_align(source, target)
        # After alignment, source should match target closely
        np.testing.assert_allclose(aligned, target, atol=1e-4)
        assert converged

    def test_icp_pure_rotation(self):
        """Source rotated 45° about z → ICP recovers rotation within tolerance."""
        rng = np.random.default_rng(42)
        target = rng.standard_normal((40, 3))
        theta = np.pi / 4  # 45 degrees
        R_gt = _rotation_z(theta)
        source = (R_gt @ target.T).T
        aligned, R, t, n_iters, converged = icp_align(source, target, max_iterations=100)
        # Aligned points should be close to the original target
        np.testing.assert_allclose(aligned, target, atol=1e-3)

    def test_icp_returns_correct_shapes(self):
        """Return shapes: aligned (N,3), R (3,3), t (3,), n_iters int, converged bool."""
        source = _make_cloud(15, seed=10)
        target = _make_cloud(20, seed=11)
        aligned, R, t, n_iters, converged = icp_align(source, target)
        assert aligned.shape == source.shape
        assert R.shape == (3, 3)
        assert t.shape == (3,)
        assert isinstance(n_iters, int)
        assert isinstance(converged, bool)

    def test_icp_convergence_flag(self):
        """max_iterations=1 must not report convergence on a shifted cloud."""
        target = _make_cloud(20, seed=5)
        source = target + np.array([5.0, 5.0, 5.0])
        _, _, _, n_iters, converged = icp_align(source, target, max_iterations=1)
        assert n_iters == 1
        # With only one iteration on a large shift, it should not have converged
        assert not converged


# ---------------------------------------------------------------------------
# Chamfer distance tests
# ---------------------------------------------------------------------------

class TestChamferDistance:
    def test_chamfer_identical_clouds_is_zero(self):
        cloud = _make_cloud(25)
        assert chamfer_distance(cloud, cloud) == pytest.approx(0.0, abs=1e-9)

    def test_chamfer_shifted_cloud(self):
        """Chamfer distance between a single point and its shifted copy equals d."""
        d = 3.0
        A = np.array([[0.0, 0.0, 0.0]])
        B = np.array([[d, 0.0, 0.0]])
        cd = chamfer_distance(A, B)
        assert cd == pytest.approx(d, rel=1e-9)

    def test_chamfer_is_symmetric(self):
        """chamfer(A, B) == chamfer(B, A)."""
        A = _make_cloud(20, seed=3)
        B = _make_cloud(25, seed=4)
        assert chamfer_distance(A, B) == pytest.approx(chamfer_distance(B, A), rel=1e-8)


# ---------------------------------------------------------------------------
# Point-to-surface tests
# ---------------------------------------------------------------------------

class TestPointToSurface:
    def test_p2s_no_faces_equals_nearest_neighbor(self):
        """Without faces, p2s equals the one-way nearest-neighbor distance."""
        points = _make_cloud(20, seed=8)
        ref = _make_cloud(30, seed=9)
        p2s = point_to_surface_distance(points, ref, reference_faces=None)
        # Manually compute nearest-neighbor
        diffs = points[:, np.newaxis, :] - ref[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        expected = float(np.mean(np.min(dists, axis=1)))
        assert p2s == pytest.approx(expected, rel=1e-7)


# ---------------------------------------------------------------------------
# SurfaceError integration tests
# ---------------------------------------------------------------------------

class TestSurfaceError:
    def test_surface_error_identical_clouds(self):
        """Identical reconstructed and reference → chamfer ≈ 0, hausdorff ≈ 0."""
        cloud = _make_cloud(30)
        result = surface_error_mm(cloud, cloud, run_icp=False)
        assert result.chamfer_mm == pytest.approx(0.0, abs=1e-6)
        assert result.hausdorff_mm == pytest.approx(0.0, abs=1e-6)

    def test_surface_error_without_icp(self):
        """run_icp=False leaves ICP fields at defaults."""
        rec = _make_cloud(20, seed=12)
        ref = _make_cloud(20, seed=13)
        result = surface_error_mm(rec, ref, run_icp=False)
        assert result.icp_iterations == 0
        assert result.icp_converged is False

    def test_surface_error_n_points_correct(self):
        """n_reconstructed and n_reference reflect input sizes."""
        rec = _make_cloud(17)
        ref = _make_cloud(23)
        result = surface_error_mm(rec, ref, run_icp=False)
        assert result.n_reconstructed == 17
        assert result.n_reference == 23

    def test_surface_error_run_icp_false(self):
        """With run_icp=False the returned SurfaceError is a frozen dataclass."""
        rec = _make_cloud(10)
        ref = _make_cloud(10)
        result = surface_error_mm(rec, ref, run_icp=False)
        assert isinstance(result, SurfaceError)
        # frozen — mutation must raise
        with pytest.raises((AttributeError, TypeError)):
            result.chamfer_mm = 999.0  # type: ignore[misc]

    def test_surface_error_with_icp_enabled(self):
        """run_icp=True should run ICP and report non-zero iterations on shifted clouds."""
        target = _make_cloud(20, seed=7)
        source = target + np.array([0.5, 0.0, 0.0])  # small shift
        result = surface_error_mm(source, target, run_icp=True)
        assert result.icp_iterations > 0
        assert isinstance(result.icp_converged, bool)
        assert result.chamfer_mm >= 0.0

    def test_surface_error_with_reference_faces(self):
        """Providing reference_faces uses triangle projection instead of NN."""
        # Unit triangle in z=0 plane
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        points = np.array([[0.1, 0.1, 1.0]])  # point above triangle
        result = surface_error_mm(points, verts, reference_faces=faces, run_icp=False)
        # point-to-surface should be ~1.0 (perpendicular distance to the plane)
        assert result.point_to_surface_mm == pytest.approx(1.0, abs=1e-5)
        assert isinstance(result, SurfaceError)


# ---------------------------------------------------------------------------
# _point_to_triangle_distance tests
# ---------------------------------------------------------------------------

class TestSurfaceErrorEmptyClouds:
    """surface_error_mm must fast-fail (raise) on empty clouds, not return NaN."""

    def test_empty_reconstructed_raises(self):
        rec = np.empty((0, 3))
        ref = _make_cloud(10)
        with pytest.raises(ValueError, match="non-empty"):
            surface_error_mm(rec, ref)

    def test_empty_reference_raises(self):
        rec = _make_cloud(10)
        ref = np.empty((0, 3))
        with pytest.raises(ValueError, match="non-empty"):
            surface_error_mm(rec, ref)

    def test_both_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            surface_error_mm(np.empty((0, 3)), np.empty((0, 3)))


class TestPointToTriangleDistanceEmpty:
    """_point_to_triangle_distance must raise ValueError on empty points or faces."""

    def test_empty_points_raises(self):
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        with pytest.raises(ValueError, match="non-empty"):
            _point_to_triangle_distance(np.empty((0, 3)), verts, faces)

    def test_empty_faces_raises(self):
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.empty((0, 3), dtype=int)
        points = np.array([[0.1, 0.1, 1.0]])
        with pytest.raises(ValueError, match="non-empty"):
            _point_to_triangle_distance(points, verts, faces)


class TestPointToTriangleDistance:
    def _unit_triangle_above(self):
        """Unit triangle in z=0 plane, with a point directly above it."""
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        return verts, faces

    def test_point_above_triangle_gives_perpendicular_distance(self):
        """A point directly above an interior triangle point → distance = height."""
        verts, faces = self._unit_triangle_above()
        # Point at (0.1, 0.1, 2.0) → projects to (0.1, 0.1, 0.0) inside triangle
        points = np.array([[0.1, 0.1, 2.0]])
        dist = _point_to_triangle_distance(points, verts, faces)
        assert dist == pytest.approx(2.0, abs=1e-5)

    def test_multiple_points_mean_is_returned(self):
        """Function returns the mean distance across all points."""
        verts, faces = self._unit_triangle_above()
        # Two points: one at height 1.0 and one at height 3.0 → mean = 2.0
        points = np.array([[0.1, 0.1, 1.0], [0.1, 0.1, 3.0]])
        dist = _point_to_triangle_distance(points, verts, faces)
        assert dist == pytest.approx(2.0, abs=1e-4)

    def test_degenerate_triangle_falls_back_to_nearest_vertex(self):
        """Degenerate (zero-area) triangle falls back to nearest vertex distance."""
        # All three vertices at the same point → denom ≈ 0
        verts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        points = np.array([[3.0, 4.0, 0.0]])
        dist = _point_to_triangle_distance(points, verts, faces)
        # Nearest vertex is at origin → distance = 5.0
        assert dist == pytest.approx(5.0, abs=1e-5)

    def test_point_outside_triangle_clamped_to_edge(self):
        """A point projected outside the triangle (s+t>1) is clamped to hypotenuse."""
        verts, faces = self._unit_triangle_above()
        # Point far along x-axis → s >> 1, t=0, s+t > 1 → hypotenuse clamping
        points = np.array([[5.0, 0.0, 1.0]])
        dist = _point_to_triangle_distance(points, verts, faces)
        assert dist > 1.0

    def test_point_with_negative_s_barycentric(self):
        """Point projected to negative-s side clamps s to 0 (line 212 branch)."""
        verts, faces = self._unit_triangle_above()
        # Project onto (-1.0, 0.5, 0.0) → s=-1.0<0, clamped to 0; t=0.5; closest=(0,0.5,0)
        points = np.array([[-1.0, 0.5, 2.0]])
        dist = _point_to_triangle_distance(points, verts, faces)
        import math
        expected = math.sqrt(1.0 + 0.0 + 4.0)  # |(-1,0.5,2) - (0,0.5,0)| = sqrt(5)
        assert dist == pytest.approx(expected, abs=1e-5)

    def test_point_with_negative_t_barycentric(self):
        """Point projected to negative-t side clamps t to 0 (line 214 branch)."""
        verts, faces = self._unit_triangle_above()
        # Project onto (0.5, -1.0, 0.0) → s=0.5; t=-1.0<0, clamped to 0; closest=(0.5,0,0)
        points = np.array([[0.5, -1.0, 0.0]])
        dist = _point_to_triangle_distance(points, verts, faces)
        assert dist == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# point_to_surface_distance with faces
# ---------------------------------------------------------------------------

class TestPointToSurfaceWithFaces:
    def test_with_faces_uses_triangle_projection(self):
        """When reference_faces is provided, uses barycentric projection."""
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        points = np.array([[0.1, 0.1, 1.5]])
        dist = point_to_surface_distance(points, verts, reference_faces=faces)
        assert dist == pytest.approx(1.5, abs=1e-5)

    def test_none_faces_and_faces_give_same_result_for_on_surface_point(self):
        """For a point on the surface, both code paths should give ≈0 distance."""
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        # Point on the surface at centroid
        points = np.array([[1.0 / 3, 1.0 / 3, 0.0]])
        dist_with_faces = point_to_surface_distance(points, verts, reference_faces=faces)
        dist_no_faces = point_to_surface_distance(points, verts, reference_faces=None)
        # Both should be small; faces version should be more accurate (≈0)
        assert dist_with_faces == pytest.approx(0.0, abs=1e-5)
        assert dist_no_faces >= 0.0


# ---------------------------------------------------------------------------
# Large-cloud chunked paths (lines 97-122, 241-252, 292-300)
# ---------------------------------------------------------------------------

class TestChunkedPaths:
    """Cover the >10_000_000 product chunked computation paths."""

    def test_chamfer_distance_chunked(self):
        """Lines 97-122: chamfer_distance uses chunked code when source * target > 10M."""
        rng = np.random.default_rng(0)
        # 1001 * 10001 = 10,012,001 > 10,000,000
        source = rng.standard_normal((1001, 3)).astype(np.float32)
        target = rng.standard_normal((10001, 3)).astype(np.float32)
        d = chamfer_distance(source, target)
        assert d > 0.0

    def test_point_to_surface_distance_chunked(self):
        """Lines 241-252: point_to_surface_distance uses chunked NN when faces=None and > 10M."""
        rng = np.random.default_rng(1)
        # 1001 * 10001 = 10,012,001 > 10,000,000
        points = rng.standard_normal((1001, 3)).astype(np.float32)
        reference = rng.standard_normal((10001, 3)).astype(np.float32)
        d = point_to_surface_distance(points, reference, reference_faces=None)
        assert d >= 0.0

    def test_surface_error_mm_hausdorff_chunked(self):
        """Lines 292-300: surface_error_mm uses chunked Hausdorff when cloud > 10M."""
        rng = np.random.default_rng(2)
        # 1001 * 10001 = 10,012,001 > 10,000,000
        rec = rng.standard_normal((1001, 3)).astype(np.float32)
        ref = rng.standard_normal((10001, 3)).astype(np.float32)
        err = surface_error_mm(rec, ref, run_icp=False)
        assert err.hausdorff_mm >= 0.0
        assert err.chamfer_mm >= 0.0


# ---------------------------------------------------------------------------
# Geometry-correctness + edge-case guards (audit fixes)
# ---------------------------------------------------------------------------

class TestSurfaceErrorGeometryFixes:
    def test_chamfer_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            chamfer_distance(np.empty((0, 3)), _make_cloud(5))
        with pytest.raises(ValueError, match="non-empty"):
            chamfer_distance(_make_cloud(5), np.empty((0, 3)))

    def test_point_to_surface_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            point_to_surface_distance(np.empty((0, 3)), _make_cloud(5))
        with pytest.raises(ValueError, match="non-empty"):
            point_to_surface_distance(_make_cloud(5), np.empty((0, 3)))

    def test_surface_error_raises_on_non_finite_coords(self):
        # A cloud with NaN/inf rows must fast-fail rather than being silently
        # filtered: surfacing the bad input beats masking it.
        ref = _make_cloud(20, seed=1)
        rec_nan = np.vstack([ref, [[np.nan, 0.0, 0.0]]])
        with pytest.raises(ValueError, match="finite"):
            surface_error_mm(rec_nan, ref, run_icp=True)
        rec_inf = np.vstack([ref, [[np.inf, 1.0, 2.0]]])
        with pytest.raises(ValueError, match="finite"):
            surface_error_mm(rec_inf, ref, run_icp=True)
        # Non-finite in the reference cloud must also raise.
        ref_bad = np.vstack([ref, [[np.nan, 0.0, 0.0]]])
        with pytest.raises(ValueError, match="finite"):
            surface_error_mm(ref, ref_bad, run_icp=True)

    def test_hausdorff_is_symmetric(self):
        # ref has a far-away point that rec does not cover. A one-sided rec->ref
        # Hausdorff would miss it; the symmetric version must capture the gap.
        rec = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ref = np.vstack([rec, [[100.0, 100.0, 100.0]]])
        result = surface_error_mm(rec, ref, run_icp=False)
        assert result.hausdorff_mm > 100.0

    def test_point_to_surface_uses_edge_segment_not_infinite_line(self):
        # Single triangle in the z=0 plane; query point far beyond the v1-v2
        # edge in the triangle plane. The true closest point is the vertex/edge
        # endpoint, not the projection onto the infinite edge line. The wrong
        # (old) clamping understated this distance.
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        # Point beyond vertex (1,0,0) along +x, in-plane: closest point is (1,0,0).
        p = np.array([[5.0, 0.0, 0.0]])
        d = _point_to_triangle_distance(p, verts, faces)
        assert d == pytest.approx(4.0, abs=1e-6)

    def test_closest_point_on_degenerate_segment_returns_endpoint(self):
        # Zero-length segment (q0 == q1): the closest point is the endpoint
        # itself, with no division by the zero segment length.
        q = np.array([1.0, 2.0, 3.0])
        out = _closest_point_on_segment(np.array([9.0, 9.0, 9.0]), q, q.copy())
        assert np.allclose(out, q)

    def test_closest_point_on_segment_clamps_to_endpoints(self):
        q0 = np.array([0.0, 0.0, 0.0])
        q1 = np.array([2.0, 0.0, 0.0])
        # Beyond q1 -> clamps to q1; before q0 -> clamps to q0; middle -> exact.
        assert np.allclose(_closest_point_on_segment(np.array([5.0, 1.0, 0.0]), q0, q1), q1)
        assert np.allclose(_closest_point_on_segment(np.array([-5.0, 1.0, 0.0]), q0, q1), q0)
        assert np.allclose(_closest_point_on_segment(np.array([1.0, 9.0, 0.0]), q0, q1), [1.0, 0.0, 0.0])


class TestChunkedPathsMatchBruteforce:
    """The >10M-pair chunked branches must equal the direct computation."""

    @staticmethod
    def _chamfer_bruteforce(s, t):
        d = np.sqrt(((s[:, None, :] - t[None, :, :]) ** 2).sum(2))
        return (d.min(1).mean() + d.min(0).mean()) / 2.0

    def test_chamfer_chunked_equals_bruteforce(self):
        rng = np.random.default_rng(0)
        s = rng.standard_normal((1001, 3))
        t = rng.standard_normal((10001, 3))  # 1001*10001 > 10M -> chunked path
        assert chamfer_distance(s, t) == pytest.approx(self._chamfer_bruteforce(s, t), rel=1e-6)

    def test_point_to_surface_chunked_equals_bruteforce(self):
        rng = np.random.default_rng(1)
        pts = rng.standard_normal((1001, 3))
        ref = rng.standard_normal((10001, 3))
        d = np.sqrt(((pts[:, None, :] - ref[None, :, :]) ** 2).sum(2))
        expected = float(np.min(d, axis=1).mean())
        assert point_to_surface_distance(pts, ref) == pytest.approx(expected, rel=1e-6)

    def test_hausdorff_chunked_equals_bruteforce(self):
        rng = np.random.default_rng(2)
        rec = rng.standard_normal((1001, 3))
        ref = rng.standard_normal((10001, 3))
        d = np.sqrt(((rec[:, None, :] - ref[None, :, :]) ** 2).sum(2))
        expected = max(float(np.max(np.min(d, axis=1))), float(np.max(np.min(d, axis=0))))
        result = surface_error_mm(rec, ref, run_icp=False)
        assert result.hausdorff_mm == pytest.approx(expected, rel=1e-6)


def test_icp_skipped_for_fewer_than_three_points():
    # run_icp=True but only 2 points: the >=3 guard fires, ICP is skipped, and
    # the metrics are still computed on the raw (un-aligned) clouds.
    rec = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ref = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    r = surface_error_mm(rec, ref, run_icp=True)
    assert r.icp_iterations == 0
    assert r.icp_converged is False
    assert r.chamfer_mm == pytest.approx(0.0, abs=1e-9)
    assert not np.isnan(r.hausdorff_mm)
