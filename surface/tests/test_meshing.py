"""Tests for screened-Poisson surface refinement (meshing.py)."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from dentalmapcert.meshing import poisson_surface_reconstruction


def _sphere_points(n: int = 4000, radius: float = 10.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * radius


def test_too_few_points_raises():
    pts = np.zeros((5, 3))
    with pytest.raises(ValueError, match="at least 30 points"):
        poisson_surface_reconstruction(pts)


def test_wrong_shape_raises():
    pts = np.zeros((50, 2))
    with pytest.raises(ValueError, match=r"\(N, 3\) array"):
        poisson_surface_reconstruction(pts)


def test_invalid_depth_raises():
    pytest.importorskip("open3d", reason="open3d not installed")
    with pytest.raises(ValueError, match="depth must be a positive integer"):
        poisson_surface_reconstruction(_sphere_points(), depth=0)


def test_open3d_unavailable_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)
    pts = _sphere_points(100)
    with pytest.raises(RuntimeError, match="open3d is required"):
        poisson_surface_reconstruction(pts)


def test_refines_a_noisy_sphere_closer_to_truth():
    """Screened Poisson on a noisy sphere returns points closer to the true
    sphere surface than the raw noisy input (the denoising property)."""
    pytest.importorskip("open3d", reason="open3d not installed")
    import open3d as o3d

    radius = 10.0
    clean = _sphere_points(6000, radius=radius, seed=1)
    rng = np.random.default_rng(2)
    noisy = clean + rng.normal(0, 0.3, clean.shape)

    refined = poisson_surface_reconstruction(noisy, depth=8, normal_radius=2.0)
    assert refined.ndim == 2 and refined.shape[1] == 3
    assert len(refined) > 100

    # Distance to the true sphere surface = |‖p‖ - radius|.
    raw_err = float(np.abs(np.linalg.norm(noisy, axis=1) - radius).mean())
    ref_err = float(np.abs(np.linalg.norm(refined, axis=1) - radius).mean())
    assert ref_err < raw_err, f"Poisson should denoise: raw={raw_err:.4f} refined={ref_err:.4f}"


def test_density_crop_disabled_when_quantile_zero():
    pytest.importorskip("open3d", reason="open3d not installed")
    refined = poisson_surface_reconstruction(_sphere_points(4000), depth=7, density_quantile=0.0)
    assert refined.ndim == 2 and refined.shape[1] == 3
    assert len(refined) > 100


def test_empty_poisson_mesh_raises(monkeypatch):
    """If Poisson yields a mesh with no triangles, fast-fail with RuntimeError."""
    o3d = pytest.importorskip("open3d", reason="open3d not installed")
    from unittest.mock import patch

    pts = _sphere_points(200)
    empty_mesh = o3d.geometry.TriangleMesh()  # no triangles
    empty_dens = o3d.utility.DoubleVector([])
    with patch.object(
        o3d.geometry.TriangleMesh,
        "create_from_point_cloud_poisson",
        return_value=(empty_mesh, empty_dens),
    ):
        with pytest.raises(RuntimeError, match="empty mesh"):
            poisson_surface_reconstruction(pts)
