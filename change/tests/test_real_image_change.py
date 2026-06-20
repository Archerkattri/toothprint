"""Tests for the pixel-level change renderer and real-image detector eval."""
from __future__ import annotations

import numpy as np
import pytest

from dcc.benchmark.real_image_eval import (
    acquire,
    evaluate_real_image_pairs,
    scorable_teeth,
    summarize_real_image_decisions,
)
from dcc.certificate.conformal import AsymmetricConformalInterval
from dcc.perturb.image_change import render_crestal_change


# ---------------------------------------------------------------------------
# render_crestal_change
# ---------------------------------------------------------------------------

def test_render_zero_delta_is_identity():
    img = np.random.default_rng(0).integers(0, 255, (40, 40, 3), dtype=np.uint8)
    out = render_crestal_change(img, [10, 10], [10, 30], 0.0)
    assert np.array_equal(out, img)
    assert out is not img  # a copy


def test_render_moves_a_bright_marker_along_bone_vector():
    # Bright dot at the crest; after a downward shift the brightness mass moves down.
    img = np.zeros((80, 80), dtype=np.uint8)
    cej_mid = [40, 20]
    crest_mid = [40, 50]
    img[48:53, 38:43] = 255  # marker at the crest
    out = render_crestal_change(img, cej_mid, crest_mid, 10.0, sigma_along=12.0, width_px=24.0)
    # Centre-of-mass of bright pixels should move in +y (down, away from cej).
    ys_before = np.argwhere(img > 50)[:, 0].mean()
    ys_after = np.argwhere(out > 50)[:, 0].mean()
    assert ys_after > ys_before + 2.0


def test_render_rejects_bad_ndim():
    with pytest.raises(ValueError, match="2D or 3D"):
        render_crestal_change(np.zeros((4, 4, 3, 2)), [1, 1], [1, 2], 5.0)


def test_render_rejects_coincident_points():
    with pytest.raises(ValueError, match="coincide"):
        render_crestal_change(np.zeros((10, 10), dtype=np.uint8), [5, 5], [5, 5], 4.0)


def test_render_window_clamped_to_empty_returns_copy():
    # Crest far outside the image with tiny sigma -> the 3-sigma window misses the
    # image entirely, so the function returns an unchanged copy.
    img = np.full((10, 10), 7, dtype=np.uint8)
    out = render_crestal_change(img, [-100, -100], [-100, -90], 5.0,
                                sigma_along=1.0, width_px=2.0)
    assert np.array_equal(out, img)


# ---------------------------------------------------------------------------
# acquire + scorable_teeth
# ---------------------------------------------------------------------------

def test_acquire_zero_noise_is_identity_copy():
    img = np.full((8, 8, 3), 100, dtype=np.uint8)
    out = acquire(img, np.random.default_rng(0), 0.0)
    assert np.array_equal(out, img) and out is not img


def test_acquire_adds_noise():
    img = np.full((8, 8, 3), 100, dtype=np.uint8)
    out = acquire(img, np.random.default_rng(0), 10.0)
    assert not np.array_equal(out, img)
    assert out.dtype == np.uint8


def test_scorable_teeth_filters_incomplete():
    class _Rec:
        annotation_dict = {"teeth": [
            {"tooth_id": "1", "cej": [[1, 1]], "crest_line": [[1, 5]]},
            {"tooth_id": "2", "cej": [], "crest_line": [[1, 5]]},   # no cej
            {"tooth_id": "3", "cej": [[1, 1]], "crest_line": []},   # no crest
        ]}
    teeth = scorable_teeth(_Rec())
    assert [t["tooth_id"] for t in teeth] == ["1"]


# ---------------------------------------------------------------------------
# evaluate_real_image_pairs (with a deterministic fake detector)
# ---------------------------------------------------------------------------

class _GtTooth(dict):
    pass


class _Rec:
    def __init__(self, image_id, teeth):
        self.image_id = image_id
        self.image_path = f"{image_id}.png"
        self.annotation_dict = {"teeth": teeth}


class _FakeDetector:
    """Returns the GT-ish landmarks read straight from the crop's known layout.

    To make the change measurable, this fake reports the crest at a position that
    tracks the actual pixel content: it returns the centre-of-mass of bright
    pixels in the crop for the crest, and a fixed CEJ. That way the rendered
    crest shift produces a real measured delta.
    """

    def predict_crop(self, crop_rgb):
        gray = crop_rgb if crop_rgb.ndim == 2 else crop_rgb.mean(axis=2)
        h, w = gray.shape[:2]
        bright = np.argwhere(gray > 50)
        if len(bright):
            cy, cx = bright[:, 0].mean(), bright[:, 1].mean()
        else:
            cy, cx = h / 2.0, w / 2.0
        cej = [w * 0.5, h * 0.2]
        crest = [float(cx), float(cy)]
        # order: cej_left, cej_right, crest_mesial, crest_distal, apex
        return [cej, cej, crest, crest, [w * 0.5, h * 0.9]]


def _img_with_crest_marker(tmp_path, name, crest_xy):
    from PIL import Image
    arr = np.zeros((120, 120, 3), dtype=np.uint8)
    x, y = crest_xy
    arr[y - 3:y + 3, x - 3:x + 3] = 255
    p = tmp_path / f"{name}.png"
    Image.fromarray(arr).save(p)
    return arr


def test_evaluate_real_image_pairs_progressed_score_tracks_delta(tmp_path, monkeypatch):
    crest_xy = (60, 70)
    arr = _img_with_crest_marker(tmp_path, "img0", crest_xy)
    tooth = {"tooth_id": "11", "cej": [[55.0, 40.0], [65.0, 40.0]],
             "crest_line": [[57.0, 70.0], [63.0, 70.0]]}
    rec = _Rec("img0", [tooth])

    rows = evaluate_real_image_pairs(
        [rec], _FakeDetector(), delta_px=12.0, acq_noise_std=0.0,
        max_teeth_per_image=3, seed=0, image_loader=lambda _p: arr,
    )
    labels = sorted({r["label"] for r in rows})
    assert labels == ["progressed", "stable"]
    stable = [r["score"] for r in rows if r["label"] == "stable"][0]
    prog = [r["score"] for r in rows if r["label"] == "progressed"][0]
    # Stable: same pixels -> ~0 change. Progressed: crest rendered downward -> >0.
    assert abs(stable) < 2.0
    assert prog > stable + 3.0


def test_evaluate_skips_record_with_no_scorable_teeth(tmp_path):
    rec = _Rec("empty", [{"tooth_id": "1", "cej": [], "crest_line": []}])
    rows = evaluate_real_image_pairs(
        [rec], _FakeDetector(), delta_px=10.0,
        image_loader=lambda _p: np.zeros((30, 30, 3), dtype=np.uint8),
    )
    assert rows == []


def test_evaluate_default_pil_loader_path(tmp_path):
    # Exercises the default PIL image_loader branch (no custom loader passed).
    crest_xy = (60, 70)
    _img_with_crest_marker(tmp_path, "img1", crest_xy)
    tooth = {"tooth_id": "11", "cej": [[55.0, 40.0], [65.0, 40.0]],
             "crest_line": [[57.0, 70.0], [63.0, 70.0]]}

    class _R:
        image_id = "img1"
        image_path = str(tmp_path / "img1.png")
        annotation_dict = {"teeth": [tooth]}

    rows = evaluate_real_image_pairs([_R()], _FakeDetector(), delta_px=10.0, acq_noise_std=0.0)
    assert len(rows) == 2


def test_evaluate_skips_when_detector_returns_degenerate_crop(tmp_path):
    # A tooth whose box collapses (image width 1) -> predict_tooth returns None.
    tooth = {"tooth_id": "1", "cej": [[0.0, 10.0], [0.0, 12.0]],
             "crest_line": [[0.0, 20.0], [0.0, 22.0]]}
    rec = _Rec("x", [tooth])
    rows = evaluate_real_image_pairs(
        [rec], _FakeDetector(), delta_px=10.0,
        image_loader=lambda _p: np.zeros((40, 1, 3), dtype=np.uint8),
    )
    assert rows == []


# ---------------------------------------------------------------------------
# summarize_real_image_decisions
# ---------------------------------------------------------------------------

def test_evaluate_skips_when_followup_detection_fails(tmp_path, monkeypatch):
    """Line 88-89: base detection succeeds but a follow-up returns None -> skip."""
    import dcc.benchmark.real_image_eval as mod

    calls = {"n": 0}

    def _fake_predict_tooth(detector, img, tooth):
        calls["n"] += 1
        return {"tooth_id": "11", "cej": [[1.0, 1.0]], "crest_line": [[1.0, 5.0]]} if calls["n"] == 1 else None

    monkeypatch.setattr(mod, "predict_tooth", _fake_predict_tooth)
    tooth = {"tooth_id": "11", "cej": [[55.0, 40.0], [65.0, 40.0]],
             "crest_line": [[57.0, 70.0], [63.0, 70.0]]}
    rec = _Rec("x", [tooth])
    rows = mod.evaluate_real_image_pairs(
        [rec], _FakeDetector(), delta_px=10.0,
        image_loader=lambda _p: np.zeros((120, 120, 3), dtype=np.uint8),
    )
    assert rows == []


def test_evaluate_skips_when_scoring_raises(tmp_path, monkeypatch):
    """Lines 96-97: empty predicted crest -> scalar_change_score KeyError -> skip."""
    import dcc.benchmark.real_image_eval as mod

    def _fake_predict_tooth(detector, img, tooth):
        return {"tooth_id": "11", "cej": [[1.0, 1.0]], "crest_line": []}  # no crest -> unscorable

    monkeypatch.setattr(mod, "predict_tooth", _fake_predict_tooth)
    tooth = {"tooth_id": "11", "cej": [[55.0, 40.0], [65.0, 40.0]],
             "crest_line": [[57.0, 70.0], [63.0, 70.0]]}
    rec = _Rec("x", [tooth])
    rows = mod.evaluate_real_image_pairs(
        [rec], _FakeDetector(), delta_px=10.0,
        image_loader=lambda _p: np.zeros((120, 120, 3), dtype=np.uint8),
    )
    assert rows == []


class _FixedConformal:
    """predict(score) -> (score-1, score+1); lets us force each decision branch."""

    def predict(self, score):
        return (score - 1.0, score + 1.0)


def test_summarize_covers_fp_and_uncertain_branches():
    rows = [
        {"label": "stable", "score": 100.0},      # tau<lo -> progressed -> fp
        {"label": "stable", "score": 15.0},       # tau in [lo,hi] -> uncertain
        {"label": "progressed", "score": 15.0},   # uncertain
        {"label": "progressed", "score": 100.0},  # progressed -> tp
    ]
    summary = summarize_real_image_decisions(rows, _FixedConformal(), tau=15.0)
    assert summary["n_stable"] == 2 and summary["n_progressed"] == 2
    assert summary["false_progression_rate"] == 0.5      # one stable flagged
    assert summary["true_change_recall"] == 0.5          # one progressed flagged


def test_summarize_recovers_recall_and_fpr():
    # Build calibration: stable ~0, progressed ~delta, fit conformal, then classify.
    delta = 20.0
    cal_rows, test_rows = [], []
    rng = np.random.default_rng(0)
    for i in range(60):
        cal_rows.append({"label": "stable", "score": float(rng.normal(0, 1)), "true_change": 0.0})
        cal_rows.append({"label": "progressed", "score": float(delta + rng.normal(0, 1)), "true_change": delta})
    for i in range(40):
        test_rows.append({"label": "stable", "score": float(rng.normal(0, 1)), "true_change": 0.0})
        test_rows.append({"label": "progressed", "score": float(delta + rng.normal(0, 1)), "true_change": delta})

    conformal = AsymmetricConformalInterval.fit(
        [r["score"] for r in cal_rows], [r["true_change"] for r in cal_rows], alpha=0.1,
    )
    summary = summarize_real_image_decisions(test_rows, conformal, tau=delta / 2.0)
    assert summary["n_stable"] == 40 and summary["n_progressed"] == 40
    # A clear delta well above the noise should give high recall and near-zero FPR.
    assert summary["true_change_recall"] > 0.7
    assert summary["false_progression_rate"] < 0.2
    assert 0.0 <= summary["stable_certification_rate"] <= 1.0
