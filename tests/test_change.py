import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from scipy.ndimage import gaussian_filter, shift as ndshift

from toothprint.change.certificate import ChangeCertificate, bone_vector, certify_change
from toothprint.change.conformal import CHANGED, STABLE, UNCERTAIN, ConformalCertifier
from toothprint.change.registration import (
    _patch, _subpixel_peak, fit_global_motion, measure_change, measure_change_anchored,
    measure_change_search, measure_displacement,
)


def _affine_warp(g, angle, tx, ty, scale=1.0):
    """Global repositioning: rotation about centre + translation (+ magnification)."""
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _crown_anchors(cej_c, L, u):
    p = (-u[1], u[0])  # perpendicular to the bone vector
    out = []
    for row in (0.5, 0.8):
        for s in (-1, 0, 1):
            out.append((cej_c[0] - row * L * u[0] + s * 0.4 * L * p[0],
                        cej_c[1] - row * L * u[1] + s * 0.4 * L * p[1]))
    return out


def _textured(n=160, seed=0):
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.normal(0, 1, (n, n)), 1.5)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return img.astype(np.float32)


# --- registration ----------------------------------------------------------

def test_patch_bounds():
    g = _textured(80)
    assert _patch(g, 40, 40, 20).shape == (40, 40)
    assert _patch(g, 2, 2, 20) is None


def test_subpixel_peak_refines():
    res = np.zeros((5, 5), np.float32)
    for i in range(5):
        for j in range(5):
            res[i, j] = -((j - 2.3) ** 2) - ((i - 1.7) ** 2)
    sx, sy = _subpixel_peak(res, 2, 2)
    assert 2.0 < sx < 2.6 and 1.4 < sy < 2.0


def test_subpixel_peak_border_integer():
    res = np.random.default_rng(0).normal(size=(5, 5)).astype(np.float32)
    assert _subpixel_peak(res, 0, 4) == (0.0, 4.0)


def test_measure_displacement_known_shift():
    g0 = _textured(seed=1)
    g1 = ndshift(g0, (5.0, 8.0), order=1, mode="reflect")
    (dx, dy), resp = measure_displacement(g0, g1, (80, 80), half=24, search=30)
    assert abs(dx - 8.0) < 1.0 and abs(dy - 5.0) < 1.0 and resp > 0.5


def test_measure_displacement_out_of_bounds():
    g = _textured(60)
    assert measure_displacement(g, g, (5, 5), half=20, search=30) is None


def test_measure_change_reference_cancels_global_motion():
    g0 = _textured(220, seed=2)
    ref_c, crest_c, u = (70, 90), (70, 140), (0.0, 1.0)
    g1 = ndshift(g0, (4.0, 3.0), order=1, mode="reflect")
    band = ndshift(g0, (16.0, 3.0), order=1, mode="reflect")
    yy = np.arange(g0.shape[0])[:, None]
    w = np.exp(-((yy - crest_c[1]) ** 2) / (2 * 18.0 ** 2))
    g1 = (g1 * (1 - w) + band * w).astype(np.float32)
    change, resp = measure_change(g0, g1, ref_c, crest_c, u, half=20, search=40)
    assert change > 6.0


def test_measure_change_out_of_bounds():
    g = _textured(60)
    assert measure_change(g, g, (5, 5), (5, 50), (0, 1), half=20, search=30) is None


def test_measure_change_search_finds_margin():
    g0 = _textured(240, seed=4)
    ref_c, coarse_crest, true_crest, u = (80, 80), (80, 120), (80, 150), (0.0, 1.0)
    g1 = ndshift(g0, (3.0, 2.0), order=1, mode="reflect")
    band = ndshift(g0, (17.0, 2.0), order=1, mode="reflect")
    yy = np.arange(g0.shape[0])[:, None]
    w = np.exp(-((yy - true_crest[1]) ** 2) / (2 * 16.0 ** 2))
    g1 = (g1 * (1 - w) + band * w).astype(np.float32)
    single = measure_change(g0, g1, ref_c, coarse_crest, u, half=18, search=40)
    best = measure_change_search(g0, g1, ref_c, coarse_crest, u, range(-40, 41, 8), half=18, search=40)
    assert best is not None and best[0] > 6.0 and best[0] >= single[0]


def test_measure_change_search_all_oob_none():
    g = _textured(60)
    assert measure_change_search(g, g, (5, 5), (5, 50), (0, 1), range(-40, 41, 8), half=20, search=30) is None


def _band_scene(seed=4):
    g0 = _textured(240, seed=seed)
    ref_c, coarse_crest, true_crest, u = (80, 80), (80, 120), (80, 150), (0.0, 1.0)
    g1 = ndshift(g0, (3.0, 2.0), order=1, mode="reflect")
    band = ndshift(g0, (17.0, 2.0), order=1, mode="reflect")
    yy = np.arange(g0.shape[0])[:, None]
    w = np.exp(-((yy - true_crest[1]) ** 2) / (2 * 16.0 ** 2))
    g1 = (g1 * (1 - w) + band * w).astype(np.float32)
    return g0, g1, ref_c, coarse_crest, u


def test_measure_change_search_response_gate_finds_margin():
    g0, g1, ref_c, crest, u = _band_scene()
    out = measure_change_search(g0, g1, ref_c, crest, u, range(-40, 41, 8),
                                half=18, search=40, min_response=0.3)
    assert out is not None and out[0] > 6.0 and out[1] >= 0.3


def test_measure_change_search_all_gated_returns_fallback():
    # An impossibly high gate rejects every candidate; the most-reliable one is
    # still returned (fallback), never None.
    g0, g1, ref_c, crest, u = _band_scene()
    out = measure_change_search(g0, g1, ref_c, crest, u, range(-40, 41, 8),
                                half=18, search=40, min_response=2.0)
    assert out is not None


# --- margin snapping (detector localization refinement) --------------------

def test_snap_to_margin_locks_onto_edge():
    from toothprint.change.registration import snap_to_margin
    # horizontal bone margin at y=150: bright (bone) above, dark below
    g = np.zeros((300, 120), np.float32)
    g[:150] = 200.0
    g = (g + np.random.default_rng(0).normal(0, 2, g.shape)).astype(np.float32)
    u = (0.0, 1.0)
    # detector crest 25px off the true margin -> snaps back onto y~150
    snapped = snap_to_margin(g, (60, 175), u, span=40, step=2.0)
    assert abs(snapped[1] - 150) < 4.0 and abs(snapped[0] - 60) < 1e-6


def test_snap_to_margin_no_edge_keeps_center():
    from toothprint.change.registration import snap_to_margin
    flat = np.full((80, 80), 100.0, np.float32)   # no edge -> centre is best
    assert snap_to_margin(flat, (40, 40), (0.0, 1.0), span=20) == (40.0, 40.0)


def test_snap_to_margin_center_out_of_bounds_finds_edge():
    from toothprint.change.registration import snap_to_margin
    # centre near the border (its own gradient sample is out of bounds) but the
    # span reaches the margin at y=20 and snaps to it
    g = np.zeros((40, 40), np.float32); g[:20] = 150.0
    out = snap_to_margin(g, (20, 39), (0.0, 1.0), span=40, step=2.0)
    assert 16.0 < out[1] < 24.0


# --- global-motion model (repositioning robustness) ------------------------

def test_fit_global_motion_needs_three_reliable_anchors():
    g = _textured(120, seed=3)
    # impossibly high gate drops every anchor -> None
    assert fit_global_motion(g, g, [(40, 40), (60, 60), (80, 80), (50, 70)],
                             half=15, search=20, min_response=1.5) is None
    # an out-of-bounds anchor is skipped; the remaining 2 < 3 -> None
    assert fit_global_motion(g, g, [(5, 5), (60, 60), (80, 80)], half=15, search=20) is None


def test_anchored_cancels_magnification_where_single_ref_fails():
    # 6% magnification (a projection-distance change between visits) moves the crest
    # and the crown reference by *different* amounts along the bone axis — exactly
    # what a single reference patch cannot cancel but a multi-anchor affine can.
    g0 = _textured(300, seed=11)
    g1 = _affine_warp(g0, angle=1.5, tx=8.0, ty=-5.0, scale=1.06)   # no local change
    cej_c, crest_c, u, L = (150, 150), (150, 195), (0.0, 1.0), 45.0
    anchors = _crown_anchors(cej_c, L, u)
    out = measure_change_anchored(g0, g1, anchors, crest_c, u, half=18, search=45)
    assert out is not None and abs(out[0]) < 2.0        # affine model cancels the motion
    single = measure_change(g0, g1, anchors[1], crest_c, u, half=18, search=45)
    assert single is not None and abs(single[0]) > abs(out[0]) + 2.0  # single-ref is fooled


def test_anchored_recovers_local_change_under_repositioning():
    g0 = _textured(300, seed=12)
    cej_c, crest_c, u, L = (150, 150), (150, 195), (0.0, 1.0), 45.0
    band = ndshift(g0, (13.0, 0.0), order=1, mode="reflect")   # apical crest move
    yy = np.arange(g0.shape[0])[:, None]
    wmask = np.exp(-((yy - crest_c[1]) ** 2) / (2 * 15.0 ** 2))
    changed = (g0 * (1 - wmask) + band * wmask).astype(np.float32)
    g1 = _affine_warp(changed, angle=2.0, tx=7.0, ty=-4.0, scale=1.05)  # change THEN reposition
    out = measure_change_anchored(g0, g1, _crown_anchors(cej_c, L, u), crest_c, u,
                                  half=18, search=45)
    assert out is not None and out[0] > 6.0                    # recovers it despite motion


def test_anchored_none_paths():
    g = _textured(120, seed=4)
    # < 3 anchors -> global fit None -> None
    assert measure_change_anchored(g, g, [(40, 40), (60, 60)], (60, 80), (0, 1),
                                   half=15, search=20) is None
    # crest patch out of bounds -> None
    assert measure_change_anchored(g, g, [(40, 40), (60, 60), (80, 80)], (5, 5), (0, 1),
                                   half=15, search=20) is None


# --- conformal -------------------------------------------------------------

def test_conformal_fit_interval_classify():
    rng = np.random.default_rng(0)
    pred = rng.normal(0, 1, 80)
    cert = ConformalCertifier.fit(pred, np.zeros(80), alpha=0.1)
    assert cert.q_lo > 0 and cert.q_hi > 0
    assert cert.classify(50.0, tau=2.0) == CHANGED
    assert cert.classify(0.0, tau=2.0) == STABLE
    lo, hi = cert.interval(5.0)
    assert lo < 5.0 < hi


def test_conformal_uncertain_band():
    cert = ConformalCertifier(q_lo=1.0, q_hi=1.0, alpha=0.1)
    assert cert.classify(2.0, tau=2.0) == UNCERTAIN


def test_conformal_small_n_abstains():
    cert = ConformalCertifier.fit([0.0, 1.0], [0.0, 0.0], alpha=0.1)
    assert np.isinf(cert.q_hi)


def test_conformal_fit_errors():
    with pytest.raises(ValueError, match="same shape"):
        ConformalCertifier.fit([1, 2], [1])
    with pytest.raises(ValueError, match="at least one"):
        ConformalCertifier.fit([], [])
    with pytest.raises(ValueError, match="alpha"):
        ConformalCertifier.fit([1.0], [0.0], alpha=2.0)


# --- certificate -----------------------------------------------------------

def test_bone_vector():
    assert np.allclose(bone_vector([0, 0], [0, 10]), [0, 1])


def test_bone_vector_coincident_raises():
    with pytest.raises(ValueError, match="coincide"):
        bone_vector([5, 5], [5, 5])


def test_certify_change_end_to_end():
    # image large enough for the default half=20 + search=70 window (>= 90px margin)
    g0 = _textured(260, seed=7)
    ref_c, crest_c, u = (130, 110), (130, 170), (0.0, 1.0)
    g1 = ndshift(g0, (3.0, 2.0), order=1, mode="reflect")
    band = ndshift(g0, (15.0, 2.0), order=1, mode="reflect")
    yy = np.arange(g0.shape[0])[:, None]
    w = np.exp(-((yy - crest_c[1]) ** 2) / (2 * 16.0 ** 2))
    g1 = (g1 * (1 - w) + band * w).astype(np.float32)
    cert = ConformalCertifier(q_lo=0.5, q_hi=0.5, alpha=0.1)
    out = certify_change(g0, g1, ref_c, crest_c, u, cert, tau=3.0)
    assert isinstance(out, ChangeCertificate)
    assert out.label == CHANGED and out.measured_px > 3.0
    # search variant exercises the candidate-offset branch
    out2 = certify_change(g0, g1, ref_c, crest_c, u, cert, tau=3.0, offsets=range(-16, 17, 8))
    assert out2.label == CHANGED


def test_certify_change_out_of_bounds_raises():
    g = _textured(60)
    cert = ConformalCertifier(q_lo=0.5, q_hi=0.5, alpha=0.1)
    with pytest.raises(ValueError, match="out of bounds"):
        certify_change(g, g, (5, 5), (5, 50), (0, 1), cert, tau=1.0)
