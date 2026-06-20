"""Tests for 2D landmark-constellation dental identification."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toothid.landmark_id import (  # noqa: E402
    _umeyama_2d, constellation, icp_residual_2d, residual_matrix,
)


def _ann(seed=0, n_teeth=6):
    rng = np.random.default_rng(seed)
    teeth = []
    for k in range(n_teeth):
        cx = 50 + k * 40
        teeth.append({
            "tooth_id": str(k),
            "cej": [[cx, 100.0 + rng.normal(0, 5)], [cx + 15, 100.0]],
            "crest_line": [[cx, 130.0 + rng.normal(0, 5)], [cx + 15, 130.0]],
            "apex": [[cx + 7, 200.0 + rng.normal(0, 5)]],
        })
    return {"teeth": teeth}


def test_constellation_collects_points():
    c = constellation(_ann(0, n_teeth=3))
    assert c.shape[1] == 2
    assert len(c) == 3 * 5  # 2 cej + 2 crest + 1 apex per tooth


def test_constellation_empty_raises():
    with pytest.raises(ValueError, match="no landmarks"):
        constellation({"teeth": [{"tooth_id": "1", "cej": [], "crest_line": []}]})


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(3)
    src = rng.normal(size=(20, 2))
    ang = 0.4
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = (1.7 * src @ R.T) + np.array([3.0, -2.0])
    A, t = _umeyama_2d(src, dst)
    recon = src @ A.T + t
    assert np.allclose(recon, dst, atol=1e-6)


def test_icp_genuine_low_impostor_high():
    base = constellation(_ann(1))
    rng = np.random.default_rng(9)
    # same person, re-acquired: similarity transform + small jitter
    ang = 0.1
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    genuine = (1.1 * base @ R.T) + np.array([10.0, -5.0]) + rng.normal(0, 1.0, base.shape)
    impostor = constellation(_ann(2))
    g_res = icp_residual_2d(genuine, base)
    i_res = icp_residual_2d(impostor, base)
    assert g_res < 3.0
    assert i_res > g_res


def test_icp_identical_converges_to_zero():
    c = constellation(_ann(4))
    assert icp_residual_2d(c, c) < 1e-6


def test_residual_matrix_shape_and_diagonal():
    anns = [_ann(s) for s in (5, 6, 7)]
    cons = [constellation(a) for a in anns]
    M = residual_matrix(cons, cons)
    assert M.shape == (3, 3)
    for i in range(3):
        assert M[i, i] == min(M[i])  # self is the best match
