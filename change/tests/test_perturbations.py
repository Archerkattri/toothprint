import unittest
import pytest


def _annotation() -> dict:
    return {
        "image": "case001.png",
        "teeth": [
            {
                "tooth_id": "36",
                "cej": [[10.0, 20.0], [30.0, 20.0]],
                "apex": [[20.0, 80.0]],
                "crest_line": [[10.0, 35.0], [30.0, 35.0]],
            }
        ],
    }


class PerturbationTests(unittest.TestCase):
    def test_acquisition_transform_moves_all_landmarks_and_keeps_stable_label(self):
        from dcc.perturb.acquisition import TransformParams, apply_acquisition_perturbation

        pair = apply_acquisition_perturbation(_annotation(), TransformParams(dx=2.0, dy=-1.0))

        tooth = pair.followup["teeth"][0]
        self.assertEqual(pair.label, "stable")
        self.assertEqual(tooth["cej"][0], [12.0, 19.0])
        self.assertEqual(tooth["apex"][0], [22.0, 79.0])
        self.assertEqual(tooth["crest_line"][0], [12.0, 34.0])

    def test_true_change_moves_only_crest_line(self):
        from dcc.perturb.truechange import inject_crestal_change, is_local_crest_change

        changed = inject_crestal_change(_annotation(), tooth_id="36", delta_px=5.0)
        tooth = changed.followup["teeth"][0]

        self.assertEqual(changed.label, "progressed")
        self.assertEqual(tooth["cej"][0], [10.0, 20.0])
        self.assertEqual(tooth["apex"][0], [20.0, 80.0])
        self.assertEqual(tooth["crest_line"][0], [10.0, 40.0])
        self.assertTrue(is_local_crest_change(changed.baseline, changed.followup, "36"))

    def test_inject_crestal_change_bone_level_matches_delta_for_horizontal_direction(self):
        """FIX 4: a 20px shift along a horizontal bone direction increases bone level by ~20px."""
        from dcc.perturb.truechange import inject_crestal_change
        from dcc.score.periodontal import tooth_bone_level

        # Horizontal tooth: CEJ is at y=20, crest is at y=20 (directly to the right)
        # so bone vector is (1, 0) — horizontal
        horizontal_ann = {
            "image": "case_h.png",
            "teeth": [
                {
                    "tooth_id": "1",
                    "cej": [[0.0, 20.0], [0.0, 20.0]],   # CEJ mid = (0, 20)
                    "crest_line": [[40.0, 20.0], [40.0, 20.0]],  # crest mid = (40, 20)
                    "apex": [[80.0, 20.0]],
                }
            ],
        }
        baseline_level = tooth_bone_level(horizontal_ann["teeth"][0])
        result = inject_crestal_change(horizontal_ann, tooth_id="1", delta_px=20.0)
        follow_level = tooth_bone_level(result.followup["teeth"][0])

        # Bone level should increase by ~20px along the horizontal vector
        self.assertAlmostEqual(follow_level - baseline_level, 20.0, places=5)


def test_crestal_shift_not_reproducible_by_acquisition_perturbation():
    """Injected crestal true-change must not be reproducible by any member of
    the acquisition-perturbation family (rigid translate + scale + noise).

    If any acquisition perturbation can produce the same signed change score as
    the synthetic crestal shift, the pair-builder is not a valid conformal test
    oracle — stable pairs could masquerade as progression.
    """
    import itertools
    from dcc.data.pair_builder import _inject_crestal_shift
    from dcc.perturb.acquisition import apply_acquisition_perturbation, TransformParams
    from dcc.score.periodontal import scalar_change_score

    ann = {
        "teeth": [{"tooth_id": "1",
                    "cej": [[100.0, 200.0], [150.0, 200.0]],
                    "crest_line": [[100.0, 300.0], [150.0, 300.0]],
                    "apex": [[125.0, 400.0]]}]
    }
    SHIFT = 30.0  # px — well above noise floor

    shifted = _inject_crestal_shift(ann, "1", SHIFT)
    true_change = scalar_change_score(ann, shifted)
    assert abs(true_change - SHIFT) < 0.1, "Sanity: crestal shift should equal true_change"

    # Acquisition perturbation family: rigid translate ± up to 20px, scale ± 5%
    acquisition_scores = []
    for dy in [-20, -10, -5, 0, 5, 10, 20]:
        for dx in [-20, -10, 0, 10, 20]:
            for scale in [0.95, 1.0, 1.05]:
                params = TransformParams(dx=float(dx), dy=float(dy), scale=float(scale))
                perturbed = apply_acquisition_perturbation(ann, params)
                score = scalar_change_score(perturbed.baseline, perturbed.followup)
                acquisition_scores.append(score)

    # No acquisition perturbation should produce a score >= SHIFT/2
    # (crestal shift signal must exceed the perturbation-family range)
    max_acq_score = max(abs(s) for s in acquisition_scores)
    assert max_acq_score < SHIFT / 2, (
        f"Acquisition perturbation reached score {max_acq_score:.2f} px "
        f"which is ≥ SHIFT/2={SHIFT/2:.2f}. "
        "The crestal shift is reproducible by acquisition perturbation — "
        "the test oracle is invalid."
    )


def test_per_landmark_perturbation_returns_stable_pair_with_displaced_landmarks():
    """apply_per_landmark_perturbation displaces landmark points and labels pair stable."""
    from dcc.perturb.acquisition import apply_per_landmark_perturbation

    ann = _annotation()
    pair = apply_per_landmark_perturbation(ann, dx_per_point=5.0, dy_per_point=5.0, seed=42)

    assert pair.label == "stable"
    assert pair.true_change == 0.0
    # Baseline must be an independent copy of original
    assert pair.baseline["teeth"][0]["cej"] == [[10.0, 20.0], [30.0, 20.0]]
    # Followup points must have been displaced
    followup_cej = pair.followup["teeth"][0]["cej"]
    assert followup_cej != [[10.0, 20.0], [30.0, 20.0]], "CEJ should be displaced"
    assert len(followup_cej) == 2, "Point count must be preserved"


def test_per_landmark_perturbation_reproducible_with_same_seed():
    """apply_per_landmark_perturbation produces identical output for same seed."""
    from dcc.perturb.acquisition import apply_per_landmark_perturbation

    ann = _annotation()
    pair1 = apply_per_landmark_perturbation(ann, dx_per_point=3.0, dy_per_point=3.0, seed=7)
    pair2 = apply_per_landmark_perturbation(ann, dx_per_point=3.0, dy_per_point=3.0, seed=7)

    assert pair1.followup["teeth"][0]["cej"] == pair2.followup["teeth"][0]["cej"]


def test_per_landmark_perturbation_different_seeds_give_different_output():
    """apply_per_landmark_perturbation should produce different displacements for different seeds."""
    from dcc.perturb.acquisition import apply_per_landmark_perturbation

    ann = _annotation()
    pair1 = apply_per_landmark_perturbation(ann, seed=1)
    pair2 = apply_per_landmark_perturbation(ann, seed=2)

    assert pair1.followup["teeth"][0]["cej"] != pair2.followup["teeth"][0]["cej"]


def test_per_landmark_perturbation_skips_empty_landmark_fields():
    """apply_per_landmark_perturbation ignores teeth with missing landmark fields."""
    from dcc.perturb.acquisition import apply_per_landmark_perturbation

    ann = {
        "teeth": [
            {"tooth_id": "11", "cej": [], "apex": None},  # empty / None fields
            {"tooth_id": "21"},                              # no landmark fields at all
        ]
    }
    # Should not raise; returns a stable pair
    pair = apply_per_landmark_perturbation(ann, seed=0)
    assert pair.label == "stable"


def test_exposure_perturbation_returns_stable_pair_with_metadata():
    """apply_exposure_perturbation injects exposure delta into followup metadata."""
    from dcc.perturb.acquisition import apply_exposure_perturbation

    ann = _annotation()
    pair = apply_exposure_perturbation(ann, exposure_delta=0.15)

    assert pair.label == "stable"
    assert pair.true_change == 0.0
    meta = pair.followup.get("metadata", {}).get("acquisition_perturbation", {})
    assert meta["exposure_delta"] == 0.15
    assert meta["dx"] == 0.0
    assert meta["dy"] == 0.0
    # Baseline must be untouched
    assert "metadata" not in pair.baseline


def test_exposure_perturbation_negative_delta():
    """apply_exposure_perturbation accepts negative exposure deltas."""
    from dcc.perturb.acquisition import apply_exposure_perturbation

    ann = _annotation()
    pair = apply_exposure_perturbation(ann, exposure_delta=-0.1)
    meta = pair.followup["metadata"]["acquisition_perturbation"]
    assert meta["exposure_delta"] == -0.1


def test_mean_point_raises_on_empty():
    """mean_point raises ValueError when given an empty sequence."""
    from dcc.geometry import mean_point

    with pytest.raises(ValueError, match="empty"):
        mean_point([])


def test_inject_crestal_change_unknown_tooth_raises():
    """inject_crestal_change raises KeyError when tooth_id is not in the annotation."""
    from dcc.perturb.truechange import inject_crestal_change

    with pytest.raises(KeyError):
        inject_crestal_change(_annotation(), tooth_id="99", delta_px=5.0)


def test_inject_crestal_change_missing_cej_raises():
    """No fallback: missing cej/crest_line raises ValueError instead of a Y shift."""
    from dcc.perturb.truechange import inject_crestal_change

    ann = {"teeth": [{"tooth_id": "11", "cej": [], "crest_line": [[0.0, 50.0]]}]}
    with pytest.raises(ValueError, match="needs both cej and crest_line"):
        inject_crestal_change(ann, tooth_id="11", delta_px=5.0)


def test_inject_crestal_change_zero_length_bone_vector_raises():
    """No fallback: coincident cej/crest (undefined bone vector) raises ValueError."""
    from dcc.perturb.truechange import inject_crestal_change

    ann = {
        "teeth": [
            {"tooth_id": "11", "cej": [[5.0, 5.0]], "crest_line": [[5.0, 5.0]]}
        ]
    }
    with pytest.raises(ValueError, match="bone vector undefined"):
        inject_crestal_change(ann, tooth_id="11", delta_px=3.0)


def test_is_local_crest_change_raises_on_missing_tooth():
    """is_local_crest_change raises KeyError for unknown tooth_id."""
    from dcc.perturb.truechange import is_local_crest_change

    ann = _annotation()
    with pytest.raises(KeyError):
        is_local_crest_change(ann, ann, tooth_id="99")


def test_record_change_scores_skips_teeth_missing_from_followup():
    """record_change_scores ignores baseline teeth not present in followup."""
    from dcc.score.periodontal import record_change_scores

    baseline = {
        "teeth": [
            {"tooth_id": "11", "cej": [[0.0, 0.0]], "crest_line": [[0.0, 10.0]]},
            {"tooth_id": "12", "cej": [[5.0, 0.0]], "crest_line": [[5.0, 8.0]]},
        ]
    }
    followup = {
        "teeth": [
            # tooth 11 present, tooth 12 missing
            {"tooth_id": "11", "cej": [[0.0, 0.0]], "crest_line": [[0.0, 10.0]]},
        ]
    }
    scores = record_change_scores(baseline, followup)
    # Only tooth 11 should be scored; tooth 12 skipped (line 25 coverage)
    assert "12" not in scores
    assert "11" in scores


def test_scalar_change_score_raises_when_no_overlapping_teeth():
    """scalar_change_score raises ValueError when no teeth overlap (line 39 coverage)."""
    from dcc.score.periodontal import scalar_change_score

    baseline = {"teeth": [{"tooth_id": "11", "cej": [[0.0, 0.0]], "crest_line": [[0.0, 10.0]]}]}
    followup = {"teeth": [{"tooth_id": "21", "cej": [[5.0, 0.0]], "crest_line": [[5.0, 8.0]]}]}

    with pytest.raises(ValueError, match="overlapping"):
        scalar_change_score(baseline, followup)


if __name__ == "__main__":
    unittest.main()
