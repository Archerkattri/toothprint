"""Tests for selection-aware, gallery-level identity calibration."""

from __future__ import annotations

import json

import numpy as np
import pytest

from toothprint.identity.gallery import (
    GalleryCalibration,
    decide_gallery_identity,
    evaluate_gallery_calibration,
)


def _calibration(n: int = 200, gallery_size: int = 4) -> GalleryCalibration:
    # Every row is a complete-gallery non-enrolled query. The minimum is the
    # selection statistic calibrated by GalleryCalibration, not a random pair.
    best = np.linspace(0.20, 2.19, n)
    distances = np.column_stack([
        best,
        best + 0.4,
        best + 0.8,
        best + 1.2,
    ])[:, :gallery_size]
    return GalleryCalibration.fit(
        distances,
        alpha=0.10,
        site_id="site-a",
        created_utc="2026-09-14T00:00:00Z",
    )


def test_fit_calibrates_the_complete_gallery_selection_unit():
    cal = _calibration()
    assert cal.gallery_size == 4
    assert cal.rank == 20  # floor((200 + 1) * .10)
    assert cal.threshold_distance == pytest.approx(0.20 + 19 * (1.99 / 199))
    assert cal.empirical_false_accept_rate <= cal.alpha
    assert cal.calibration_id.startswith("site-a:")
    payload = cal.to_dict()
    json.loads(json.dumps(payload))
    restored = GalleryCalibration.from_dict(payload)
    assert restored.calibration_id == cal.calibration_id
    assert restored.to_dict() == payload


def test_decision_accepts_one_identity_and_abstains_on_open_set_or_ambiguity():
    cal = _calibration()
    labels = ["patient-a", "patient-b", "patient-c", "patient-d"]

    accepted = decide_gallery_identity(
        [0.25, 1.0, 1.2, 1.4], cal, gallery_labels=labels
    )
    assert accepted.accepted is True
    assert accepted.label == "patient-a"
    assert accepted.reason == "single_gallery_identity_clears_threshold"
    assert accepted.identity_set == ("patient-a",)
    assert accepted.to_dict()["abstained"] is False

    rejected = decide_gallery_identity(
        [3.0, 3.1, 3.2, 3.3], cal, gallery_labels=labels
    )
    assert not rejected.accepted
    assert rejected.label is None
    assert rejected.reason == "best_candidate_does_not_clear_gallery_calibration"

    ambiguous = decide_gallery_identity(
        [0.25, 0.26, 1.2, 1.4], cal, gallery_labels=labels
    )
    assert not ambiguous.accepted
    assert ambiguous.reason == "multiple_gallery_identities_clear_threshold"
    assert ambiguous.identity_set == ("patient-a", "patient-b")

    margin_ambiguous = decide_gallery_identity(
        [0.25, 0.26, 1.2, 1.4], cal,
        gallery_labels=labels,
        ambiguity_margin=0.02,
    )
    assert not margin_ambiguous.accepted
    assert margin_ambiguous.reason == "multiple_gallery_identities_clear_threshold"


def test_duplicate_gallery_entries_for_one_identity_do_not_create_two_identity_set():
    cal = _calibration()
    decision = decide_gallery_identity(
        [0.25, 0.26, 1.2, 1.4], cal,
        gallery_labels=["patient-a", "patient-a", "patient-c", "patient-d"],
    )
    assert decision.accepted is True
    assert decision.label == "patient-a"
    assert decision.identity_set == ("patient-a",)


def test_heldout_evaluation_reports_gallery_fpir_and_empirical_genuine_metrics():
    cal = _calibration()
    report = evaluate_gallery_calibration(
        cal,
        impostor_distances=np.array([
            [3.0, 3.1, 3.2, 3.3],
            [0.21, 1.0, 1.1, 1.2],  # one held-out false accept
        ]),
        genuine_distances=np.array([
            [0.25, 1.0, 1.1, 1.2],
            [1.0, 0.25, 1.1, 1.2],
        ]),
        genuine_labels=["a", "b"],
        gallery_labels=["a", "b", "c", "d"],
    )
    assert report["format"] == "TOOTHPRINT_GALLERY_CALIBRATION_EVALUATION"
    assert report["n_impostor"] == 2
    assert report["false_accept_rate"] == pytest.approx(0.5)
    assert report["genuine_correct_accept_rate"] == pytest.approx(1.0)
    assert "false-accept control" in report["claim_boundary"][0]


def test_gallery_size_and_input_validation_are_hard_failures():
    cal = _calibration()
    with pytest.raises(ValueError, match="gallery size mismatch"):
        decide_gallery_identity([0.1, 0.2], cal)
    with pytest.raises(ValueError, match="finite"):
        decide_gallery_identity([np.nan, 0.2, 0.3, 0.4], cal)
    with pytest.raises(ValueError, match="ambiguity_margin"):
        decide_gallery_identity([0.1, 0.2, 0.3, 0.4], cal, ambiguity_margin=-1)
    with pytest.raises(ValueError, match="non-enrolled"):
        GalleryCalibration.fit(
            np.ones((3, 4)),
            site_id="site-a",
            created_utc="now",
            min_calibration=100,
        )


def test_small_alpha_can_safely_disable_acceptance_when_no_conformal_rank_exists():
    distances = np.ones((10, 2))
    cal = GalleryCalibration.fit(
        distances, alpha=0.01, site_id="s", created_utc="now", min_calibration=1
    )
    assert cal.rank == 0
    assert cal.threshold_distance == -np.inf
    result = decide_gallery_identity([0.0, 1.0], cal)
    assert result.accepted is False
    assert result.to_dict()["threshold_distance"] is None
