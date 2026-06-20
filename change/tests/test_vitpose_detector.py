"""Tests for the ViTPose dental landmark detector (pure, dependency-light parts)."""
from __future__ import annotations

import numpy as np
import pytest

from dcc.landmarks.vitpose_detector import (
    INPUT_H,
    INPUT_W,
    LANDMARK_NAMES,
    NUM_LANDMARKS,
    ViTPoseLandmarkDetector,
)


def test_landmark_constants():
    assert NUM_LANDMARKS == 5
    assert LANDMARK_NAMES == ("cej_left", "cej_right", "crest_mesial", "crest_distal", "apex")
    assert (INPUT_H, INPUT_W) == (256, 192)


def test_normalize_crop_shape_and_normalisation():
    pytest.importorskip("torch", reason="torch not installed")
    from dcc.landmarks.vitpose_detector import normalize_crop

    crop = (np.random.rand(80, 50, 3) * 255).astype(np.uint8)
    t = normalize_crop(crop)
    assert tuple(t.shape) == (1, 3, INPUT_H, INPUT_W)
    # ImageNet-normalised values land roughly in [-3, 3], not raw [0, 1].
    assert float(t.min()) < 0.0 < float(t.max())


def test_heatmaps_to_coords_recovers_peak():
    torch = pytest.importorskip("torch", reason="torch not installed")
    from dcc.landmarks.vitpose_detector import heatmaps_to_coords

    H, W = 64, 48
    hm = torch.zeros(1, NUM_LANDMARKS, H, W)
    peaks = [(10, 20), (30, 5), (47, 63), (1, 1), (25, 40)]  # (x, y)
    for k, (x, y) in enumerate(peaks):
        hm[0, k, y, x] = 1.0
    coords = heatmaps_to_coords(hm)
    assert tuple(coords.shape) == (1, NUM_LANDMARKS, 2)
    for k, (x, y) in enumerate(peaks):
        assert abs(float(coords[0, k, 0]) - x) <= 0.5
        assert abs(float(coords[0, k, 1]) - y) <= 0.5


def test_detector_missing_weights_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="ViTPose weights not found"):
        ViTPoseLandmarkDetector(tmp_path / "does_not_exist.pt")
