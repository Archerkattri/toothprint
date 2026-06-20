"""Tests for 3D dental biometric identification."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
o3d = pytest.importorskip("open3d")

from toothid.mesh_id import (  # noqa: E402
    compute_fpfh, enroll, identification_metrics, rank1_match,
    register_rmse, to_point_cloud,
)


def _arch(seed=0, n=1500):
    """An asymmetric 'dental-arch-like' cloud: a U-curve with cusp bumps."""
    rng = np.random.default_rng(seed)
    t = np.linspace(-1.2, 1.2, n)
    x = t * 18.0
    y = (t ** 2) * 12.0
    z = np.sin(t * 6.0) * 2.0 + rng.normal(0, 0.05, n)
    pts = np.stack([x, y, z], axis=1)
    # a couple of subject-specific bumps so arches differ
    for _ in range(3):
        c = rng.uniform(-18, 18, 3) * np.array([1, 0.6, 0.1])
        d = np.linalg.norm(pts - c, axis=1)
        pts[:, 2] += 2.0 * np.exp(-(d ** 2) / 8.0)
    return pts


def _rigid(pts, rng):
    ang = rng.uniform(-0.2, 0.2)
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return pts @ R.T + rng.uniform(-3, 3, 3)


# --- pure identification logic ---------------------------------------------

def test_rank1_match_returns_argmin():
    assert rank1_match([0.5, 0.1, 0.9]) == 1


def test_rank1_match_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        rank1_match([])


def test_identification_metrics_perfect_diagonal():
    # 3 subjects, genuine on the diagonal (low), impostor off-diagonal (high).
    rmse = np.array([[0.1, 0.8, 0.9], [0.7, 0.1, 0.8], [0.9, 0.7, 0.1]])
    labels = ["a", "b", "c"]
    m = identification_metrics(rmse, labels, labels)
    assert m["rank1_accuracy"] == 1.0
    assert m["genuine_rmse_mean"] < m["impostor_rmse_mean"]
    assert m["genuine_rmse_max"] < m["impostor_rmse_min"]
    assert m["decidability_dprime"] > 2.0


def test_identification_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        identification_metrics(np.zeros((2, 3)), ["a", "b"], ["a", "b"])


def test_identification_metrics_no_genuine_gives_nan():
    # disjoint query/gallery labels -> no genuine pairs.
    rmse = np.array([[0.3, 0.4]])
    m = identification_metrics(rmse, ["q"], ["a", "b"])
    assert np.isnan(m["genuine_rmse_mean"])
    assert m["rank1_accuracy"] == 0.0


def test_identification_metrics_zero_variance_dprime_zero():
    rmse = np.array([[0.5, 0.5]])  # genuine == impostor, zero spread
    m = identification_metrics(rmse, ["a"], ["a", "b"])
    assert m["decidability_dprime"] == 0.0


# --- open3d feature + registration pipeline --------------------------------

def test_to_point_cloud_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        to_point_cloud(np.zeros((10, 2)), 0.5)


def test_to_point_cloud_downsamples_and_has_normals():
    pcd = to_point_cloud(_arch(0), voxel_size=1.0)
    assert len(pcd.points) > 10
    assert pcd.has_normals()


def test_fpfh_has_33_dims():
    pcd = to_point_cloud(_arch(0), 1.0)
    f = compute_fpfh(pcd, 1.0)
    assert np.asarray(f.data).shape[0] == 33


def test_register_genuine_low_impostor_high_rmse():
    voxel = 1.0
    rng = np.random.default_rng(1)
    a = _arch(seed=1)
    b = _arch(seed=2)  # different subject
    q_pcd, q_f = enroll(_rigid(a, rng) + rng.normal(0, 0.05, a.shape), voxel)
    ga_pcd, ga_f = enroll(a, voxel)
    gb_pcd, gb_f = enroll(b, voxel)
    genuine_rmse, genuine_fit = register_rmse(q_pcd, q_f, ga_pcd, ga_f, voxel)
    impostor_rmse, _ = register_rmse(q_pcd, q_f, gb_pcd, gb_f, voxel)
    assert genuine_fit > 0.3
    assert genuine_rmse < impostor_rmse
