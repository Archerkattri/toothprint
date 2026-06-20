"""Tests for dcc.data.pair_builder."""

from __future__ import annotations

import math
import unittest

from dcc.data.pair_builder import (
    PairBuilderConfig,
    _add_noise,
    _inject_crestal_shift,
    _lcg,
    build_pairs,
)
from dcc.score.periodontal import scalar_change_score, tooth_bone_level


def _make_annotation(tooth_id: str = "1") -> dict:
    """Minimal DenPAR-style annotation with one tooth."""
    return {
        "teeth": [
            {
                "tooth_id": tooth_id,
                "cej": [[100.0, 200.0], [150.0, 200.0]],
                "crest_line": [[300.0, 200.0], [350.0, 200.0]],
                "apex": [[350.0, 50.0]],
            }
        ]
    }


def _make_anns(n: int) -> list:
    """Minimal record objects with annotation_dict and image_id."""

    class _Rec:
        def __init__(self, i):
            self.image_id = f"img{i}"
            self.annotation_dict = _make_annotation(str(i))

    return [_Rec(i) for i in range(n)]


class TestLcg(unittest.TestCase):
    def test_deterministic(self):
        s1, v1 = _lcg(42)
        s2, v2 = _lcg(42)
        self.assertEqual(s1, s2)
        self.assertEqual(v1, v2)

    def test_range(self):
        for seed in [0, 1, 1000, 0xFFFFFF]:
            _, v = _lcg(seed)
            self.assertGreaterEqual(v, -1.0)
            self.assertLessEqual(v, 1.0)


class TestAddNoise(unittest.TestCase):
    def test_returns_copy(self):
        ann = _make_annotation()
        noised, _ = _add_noise(ann, 0, std=1.0)
        self.assertIsNot(noised, ann)
        self.assertEqual(ann["teeth"][0]["cej"], [[100.0, 200.0], [150.0, 200.0]])

    def test_zero_std_no_change(self):
        ann = _make_annotation()
        noised, _ = _add_noise(ann, 0, std=0.0)
        self.assertEqual(noised["teeth"][0]["cej"], [[100.0, 200.0], [150.0, 200.0]])

    def test_nonzero_std_changes_coords(self):
        ann = _make_annotation()
        noised, _ = _add_noise(ann, 0, std=10.0)
        orig_pt = ann["teeth"][0]["cej"][0]
        new_pt = noised["teeth"][0]["cej"][0]
        changed = orig_pt[0] != new_pt[0] or orig_pt[1] != new_pt[1]
        self.assertTrue(changed)

    def test_deterministic(self):
        ann = _make_annotation()
        n1, _ = _add_noise(ann, 99, std=5.0)
        n2, _ = _add_noise(ann, 99, std=5.0)
        self.assertEqual(n1["teeth"][0]["cej"], n2["teeth"][0]["cej"])

    def test_different_seeds_differ(self):
        ann = _make_annotation()
        n1, _ = _add_noise(ann, 1, std=5.0)
        n2, _ = _add_noise(ann, 2, std=5.0)
        self.assertNotEqual(n1["teeth"][0]["cej"], n2["teeth"][0]["cej"])


class TestInjectCrestalShift(unittest.TestCase):
    def test_returns_copy(self):
        ann = _make_annotation()
        shifted = _inject_crestal_shift(ann, "1", 10.0)
        self.assertIsNot(shifted, ann)

    def test_bone_level_increases_by_delta(self):
        ann = _make_annotation()
        # Horizontal bone: cej_mid=(125, 200), crest_mid=(325, 200), bone=200px
        t_orig = ann["teeth"][0]
        orig_level = tooth_bone_level(t_orig)
        self.assertAlmostEqual(orig_level, 200.0, places=3)

        shifted = _inject_crestal_shift(ann, "1", 20.0)
        t_shifted = shifted["teeth"][0]
        new_level = tooth_bone_level(t_shifted)
        self.assertAlmostEqual(new_level, 220.0, places=3)

    def test_change_score_equals_delta(self):
        ann = _make_annotation()
        shifted = _inject_crestal_shift(ann, "1", 15.0)
        score = scalar_change_score(ann, shifted, tooth_id="1")
        self.assertAlmostEqual(score, 15.0, places=3)

    def test_unknown_tooth_raises_key_error(self):
        ann = _make_annotation()
        with self.assertRaises(KeyError):
            _inject_crestal_shift(ann, "999", 10.0)

    def test_inject_empty_cej_breaks_early(self):
        """_inject_crestal_shift breaks early when cej is empty (line 97)."""
        ann = {
            "teeth": [
                {
                    "tooth_id": "1",
                    "cej": [],           # empty → not cej → break at line 97
                    "crest_line": [[1.0, 1.0]],
                }
            ]
        }
        result = _inject_crestal_shift(ann, "1", 10.0)
        self.assertEqual(result["teeth"][0]["crest_line"], [[1.0, 1.0]])

    def test_inject_zero_length_bone_vector_breaks_early(self):
        """_inject_crestal_shift breaks when cej_mid == crest_mid (length < 1e-9, line 104)."""
        ann = {
            "teeth": [
                {
                    "tooth_id": "1",
                    "cej": [[5.0, 5.0]],
                    "crest_line": [[5.0, 5.0]],   # same point → dx=dy=0 → length=0 → break
                }
            ]
        }
        result = _inject_crestal_shift(ann, "1", 10.0)
        self.assertEqual(result["teeth"][0]["crest_line"], [[5.0, 5.0]])


class TestBuildPairs(unittest.TestCase):
    def test_two_pairs_per_record(self):
        records = _make_anns(5)
        pairs = build_pairs(records)
        self.assertEqual(len(pairs), 10)

    def test_labels_alternate(self):
        records = _make_anns(3)
        pairs = build_pairs(records)
        labels = [p.label for p in pairs]
        for i in range(0, len(labels), 2):
            self.assertEqual(labels[i], "stable")
            self.assertEqual(labels[i + 1], "progressed")

    def test_stable_true_change_zero(self):
        pairs = build_pairs(_make_anns(4))
        for p in pairs:
            if p.label == "stable":
                self.assertEqual(p.true_change, 0.0)

    def test_progressed_true_change_equals_shift(self):
        cfg = PairBuilderConfig(crestal_shift_px=30.0)
        pairs = build_pairs(_make_anns(4), cfg)
        for p in pairs:
            if p.label == "progressed":
                self.assertEqual(p.true_change, 30.0)

    def test_progressed_score_near_shift(self):
        cfg = PairBuilderConfig(acq_noise_std=0.0, crestal_shift_px=25.0)
        pairs = build_pairs(_make_anns(3), cfg)
        for p in pairs:
            if p.label == "progressed":
                score = scalar_change_score(p.baseline, p.followup)
                self.assertAlmostEqual(score, 25.0, places=2)

    def test_stable_score_near_zero_without_noise(self):
        cfg = PairBuilderConfig(acq_noise_std=0.0, crestal_shift_px=20.0)
        pairs = build_pairs(_make_anns(3), cfg)
        for p in pairs:
            if p.label == "stable":
                score = abs(scalar_change_score(p.baseline, p.followup))
                self.assertAlmostEqual(score, 0.0, places=6)

    def test_empty_records(self):
        self.assertEqual(build_pairs([]), [])

    def test_deterministic(self):
        records = _make_anns(4)
        p1 = build_pairs(records, PairBuilderConfig(seed=7))
        p2 = build_pairs(records, PairBuilderConfig(seed=7))
        for a, b in zip(p1, p2):
            self.assertEqual(a.baseline, b.baseline)
            self.assertEqual(a.followup, b.followup)

    def test_different_seeds_differ(self):
        records = _make_anns(4)
        p1 = build_pairs(records, PairBuilderConfig(seed=0, acq_noise_std=5.0))
        p2 = build_pairs(records, PairBuilderConfig(seed=1, acq_noise_std=5.0))
        # At least one pair baseline should differ
        any_diff = any(a.baseline != b.baseline for a, b in zip(p1, p2))
        self.assertTrue(any_diff)

    def test_skips_record_with_no_teeth(self):
        class EmptyRec:
            image_id = "empty"
            annotation_dict = {"teeth": []}

        records = [EmptyRec()] + _make_anns(2)
        pairs = build_pairs(records)
        self.assertEqual(len(pairs), 4)

    def test_skips_record_with_no_scorable_tooth(self):
        # Teeth present but none has both cej and crest_line -> no change can be
        # injected, so the record is skipped (no silently-no-op progressed pair).
        class NoScorableRec:
            image_id = "noscore"
            annotation_dict = {"teeth": [
                {"tooth_id": "9", "cej": [], "crest_line": [[1.0, 5.0]]},
                {"tooth_id": "10", "cej": [[1.0, 1.0]], "crest_line": []},
            ]}

        records = [NoScorableRec()] + _make_anns(2)
        pairs = build_pairs(records)
        self.assertEqual(len(pairs), 4)  # only the 2 scorable records produce pairs


if __name__ == "__main__":
    unittest.main()
