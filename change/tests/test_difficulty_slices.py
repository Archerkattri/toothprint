"""Tests for difficulty slices in dcc/eval/metrics.py — 8 tests."""
from __future__ import annotations
import unittest


def _make_row(
    true: str,
    decision: str,
    score: float,
    hi: float = 5.0,
    lo: float = 0.0,
) -> dict:
    return {"true": true, "decision": decision, "score": score, "hi": hi, "lo": lo}


class TestDifficultySlices(unittest.TestCase):

    def test_stable_rows_split_three_ways_by_score(self):
        """Stable rows split into low/medium/high noise at noise_low_threshold
        (3.0) and noise_high_threshold (6.0)."""
        from dcc.eval.metrics import slice_by_difficulty

        rows = [
            _make_row("stable", "stable", score=1.0),   # low noise   (< 3.0)
            _make_row("stable", "stable", score=2.9),   # low noise   (< 3.0)
            _make_row("stable", "stable", score=3.0),   # medium      (3.0 <= s < 6.0)
            _make_row("stable", "stable", score=5.9),   # medium      (3.0 <= s < 6.0)
            _make_row("stable", "stable", score=6.0),   # high noise  (>= 6.0)
            _make_row("stable", "stable", score=9.0),   # high noise  (>= 6.0)
        ]
        slices = slice_by_difficulty(rows)
        by_name = {s.name: s for s in slices}
        self.assertEqual(by_name["stable_low_noise"].summary.n, 2)
        self.assertEqual(by_name["stable_medium_noise"].summary.n, 2)
        self.assertEqual(by_name["stable_high_noise"].summary.n, 2)

    def test_progressed_rows_split_by_shift(self):
        """Progressed rows split at shift_threshold (default 10.0)."""
        from dcc.eval.metrics import slice_by_difficulty

        rows = [
            _make_row("progressed", "progressed", score=5.0),   # small shift
            _make_row("progressed", "progressed", score=9.9),   # small shift
            _make_row("progressed", "progressed", score=10.0),  # large shift
            _make_row("progressed", "progressed", score=20.0),  # large shift
        ]
        slices = slice_by_difficulty(rows)
        names = {s.name for s in slices}
        self.assertIn("progressed_small_shift", names)
        self.assertIn("progressed_large_shift", names)

        small = next(s for s in slices if s.name == "progressed_small_shift")
        large = next(s for s in slices if s.name == "progressed_large_shift")
        self.assertEqual(small.summary.n, 2)
        self.assertEqual(large.summary.n, 2)

    def test_empty_strata_not_returned(self):
        """Strata with zero rows must not appear in the result."""
        from dcc.eval.metrics import slice_by_difficulty

        # Only stable rows with low noise → only stable_low_noise stratum
        rows = [_make_row("stable", "stable", score=1.0)]
        slices = slice_by_difficulty(rows)
        names = {s.name for s in slices}
        self.assertNotIn("stable_high_noise", names)
        self.assertNotIn("progressed_small_shift", names)
        self.assertNotIn("progressed_large_shift", names)

    def test_all_stable_low_noise_one_stratum(self):
        """All rows in the same stratum → exactly one DifficultySlice returned."""
        from dcc.eval.metrics import slice_by_difficulty

        rows = [
            _make_row("stable", "stable", score=0.5),
            _make_row("stable", "stable", score=1.0),
            _make_row("stable", "stable", score=2.0),
        ]
        slices = slice_by_difficulty(rows)
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0].name, "stable_low_noise")

    def test_summary_is_DecisionSummary_instance(self):
        """Each slice's .summary should be a DecisionSummary."""
        from dcc.eval.metrics import slice_by_difficulty, DecisionSummary

        rows = [_make_row("stable", "stable", score=1.0)]
        slices = slice_by_difficulty(rows)
        self.assertIsInstance(slices[0].summary, DecisionSummary)

    def test_slice_names_are_expected(self):
        """All returned slice names must come from the known set."""
        from dcc.eval.metrics import slice_by_difficulty

        known = {"stable_low_noise", "stable_medium_noise", "stable_high_noise",
                 "progressed_small_shift", "progressed_large_shift"}
        rows = [
            _make_row("stable", "stable", score=1.0),
            _make_row("stable", "stable", score=4.0),
            _make_row("progressed", "progressed", score=5.0),
            _make_row("progressed", "progressed", score=15.0),
        ]
        slices = slice_by_difficulty(rows)
        for s in slices:
            self.assertIn(s.name, known)

    def test_no_rows_returns_empty(self):
        """Empty input → empty list."""
        from dcc.eval.metrics import slice_by_difficulty

        slices = slice_by_difficulty([])
        self.assertEqual(slices, [])

    def test_mixed_rows_five_strata(self):
        """Mixed input that populates all five strata."""
        from dcc.eval.metrics import slice_by_difficulty

        rows = [
            _make_row("stable", "stable", score=1.0),    # stable_low_noise
            _make_row("stable", "stable", score=4.0),    # stable_medium_noise
            _make_row("stable", "stable", score=7.0),    # stable_high_noise
            _make_row("progressed", "progressed", score=5.0),  # progressed_small_shift
            _make_row("progressed", "progressed", score=15.0), # progressed_large_shift
        ]
        slices = slice_by_difficulty(rows)
        self.assertEqual(len(slices), 5)
        names = {s.name for s in slices}
        self.assertEqual(names, {
            "stable_low_noise", "stable_medium_noise", "stable_high_noise",
            "progressed_small_shift", "progressed_large_shift",
        })


if __name__ == "__main__":
    unittest.main()
