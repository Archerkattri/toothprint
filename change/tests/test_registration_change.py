"""Tests for differential bone-level change measurement (registration)."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from scipy.ndimage import shift as ndshift

from dcc.score.registration_change import (
    _patch,
    _subpixel_peak,
    measure_bonelevel_change,
    measure_bonelevel_change_search,
    measure_displacement,
)


def _textured(n=160, seed=0):
    """A smooth high-frequency texture that template-matches unambiguously."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1, (n, n))
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, 1.5)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return img.astype(np.float32)


def test_patch_in_bounds_and_out():
    g = _textured(80)
    assert _patch(g, 40, 40, 20).shape == (40, 40)
    assert _patch(g, 2, 2, 20) is None      # off the top-left
    assert _patch(g, 78, 78, 20) is None    # off the bottom-right


def test_subpixel_peak_refines_toward_true_offset():
    # Build a discrete parabola peaked between samples.
    res = np.zeros((5, 5), dtype=np.float32)
    for i in range(5):
        for j in range(5):
            res[i, j] = -((j - 2.3) ** 2) - ((i - 1.7) ** 2)
    sx, sy = _subpixel_peak(res, 2, 2)
    assert 2.0 < sx < 2.6
    assert 1.4 < sy < 2.0


def test_subpixel_peak_at_border_returns_integer():
    res = np.random.default_rng(0).normal(size=(5, 5)).astype(np.float32)
    sx, sy = _subpixel_peak(res, 0, 4)  # both on the border
    assert sx == 0.0 and sy == 4.0


def test_measure_displacement_recovers_known_shift():
    g0 = _textured(160, seed=1)
    g1 = ndshift(g0, (5.0, 8.0), order=1, mode="reflect")  # (dy=5, dx=8)
    out = measure_displacement(g0, g1, (80, 80), half=24, search=30)
    assert out is not None
    (dx, dy), resp = out
    assert abs(dx - 8.0) < 1.0
    assert abs(dy - 5.0) < 1.0
    assert resp > 0.5


def test_measure_displacement_out_of_bounds_returns_none():
    g = _textured(60)
    assert measure_displacement(g, g, (5, 5), half=20, search=30) is None


def test_measure_bonelevel_change_cej_referenced_cancels_global_motion():
    g0 = _textured(220, seed=2)
    # Global shift everywhere, PLUS extra apical move localized near the crest.
    cej_c = (70, 90)
    crest_c = (70, 140)
    u = (0.0, 1.0)  # apical = +y
    g1 = ndshift(g0, (4.0, 3.0), order=1, mode="reflect")  # global (dy=4, dx=3)
    # add a further +12px y move to a band around the crest only
    band = ndshift(g0, (16.0, 3.0), order=1, mode="reflect")
    yy = np.arange(g0.shape[0])[:, None]
    wmask = np.exp(-((yy - crest_c[1]) ** 2) / (2 * 18.0 ** 2))
    g1 = (g1 * (1 - wmask) + band * wmask).astype(np.float32)
    out = measure_bonelevel_change(g0, g1, cej_c, crest_c, u, half=20, search=40)
    assert out is not None
    change, resp = out
    # CEJ sees ~4px global; crest sees ~4+12; difference ~ +12 along +y.
    assert change > 6.0


def test_measure_bonelevel_change_out_of_bounds_returns_none():
    g = _textured(60)
    assert measure_bonelevel_change(g, g, (5, 5), (5, 50), (0, 1), half=20, search=30) is None


def test_search_finds_margin_when_localization_is_off():
    g0 = _textured(240, seed=4)
    cej_c = (80, 80)
    true_crest = (80, 150)
    coarse_crest = (80, 120)  # localization 30px off the true margin
    u = (0.0, 1.0)
    g1 = ndshift(g0, (3.0, 2.0), order=1, mode="reflect")  # global motion
    band = ndshift(g0, (3.0 + 14.0, 2.0), order=1, mode="reflect")  # +14px apical near margin
    yy = np.arange(g0.shape[0])[:, None]
    wmask = np.exp(-((yy - true_crest[1]) ** 2) / (2 * 16.0 ** 2))
    g1 = (g1 * (1 - wmask) + band * wmask).astype(np.float32)
    # single patch at the coarse (wrong) crest misses most of the move; the search
    # over offsets recovers it.
    single = measure_bonelevel_change(g0, g1, cej_c, coarse_crest, u, half=18, search=40)
    best = measure_bonelevel_change_search(g0, g1, cej_c, coarse_crest, u,
                                           offsets=range(-40, 41, 8), half=18, search=40)
    assert best is not None
    assert best[0] > 6.0
    assert best[0] >= single[0]


def test_search_returns_none_when_all_candidates_out_of_bounds():
    g = _textured(60)
    assert measure_bonelevel_change_search(g, g, (5, 5), (5, 50), (0, 1),
                                           offsets=range(-40, 41, 8), half=20, search=30) is None
