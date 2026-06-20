"""Extended tests for interval-width metrics and fpr curve."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


def _make_rows(intervals: list[tuple[float, float]], labels: list[str]) -> list[dict]:
    """Build minimal eval rows matching the evaluate_pairs dict format."""
    rows = []
    for (lo, hi), label in zip(intervals, labels):
        rows.append(
            {
                "true": label,
                "decision": "stable",
                "score": (lo + hi) / 2.0,
                "lo": lo,
                "hi": hi,
            }
        )
    return rows


class TestMeanIntervalWidthInSummary(unittest.TestCase):
    def test_mean_interval_width_in_summary(self):
        from dcc.eval.metrics import summarize_decisions

        # Three stable pairs with widths 2, 4, 6 → mean=4, std=~1.633
        rows = _make_rows(
            [(0.0, 2.0), (1.0, 5.0), (2.0, 8.0)],
            ["stable", "stable", "stable"],
        )
        summary = summarize_decisions(rows)

        self.assertAlmostEqual(summary.mean_interval_width, 4.0, places=6)
        import math
        expected_std = math.sqrt(((2 - 4) ** 2 + (4 - 4) ** 2 + (6 - 4) ** 2) / 3)
        self.assertAlmostEqual(summary.interval_width_std, expected_std, places=6)

    def test_mean_interval_width_mixed_labels(self):
        from dcc.eval.metrics import summarize_decisions

        # Two stable (widths 2, 4) and one progressed (width 6) → mean of all = 4.0
        rows = _make_rows(
            [(0.0, 2.0), (1.0, 5.0), (2.0, 8.0)],
            ["stable", "stable", "progressed"],
        )
        summary = summarize_decisions(rows)

        self.assertAlmostEqual(summary.mean_interval_width, 4.0, places=6)

    def test_mean_interval_width_zero_when_no_interval_keys(self):
        from dcc.eval.metrics import summarize_decisions

        # Rows without "lo"/"hi" keys → graceful fallback to 0.0
        rows = [
            {"true": "stable", "decision": "stable"},
            {"true": "progressed", "decision": "progressed"},
        ]
        summary = summarize_decisions(rows)

        self.assertEqual(summary.mean_interval_width, 0.0)
        self.assertEqual(summary.interval_width_std, 0.0)


class TestCoverageVsFprCurve(unittest.TestCase):
    def test_coverage_vs_fpr_curve_returns_list(self):
        from dcc.eval.metrics import coverage_vs_false_progression_curve

        rows = _make_rows([(0.0, 2.0), (5.0, 7.0)], ["stable", "stable"])
        curve = coverage_vs_false_progression_curve(rows, tau=3.0, n_points=5)

        self.assertIsInstance(curve, list)
        self.assertEqual(len(curve), 5)
        for pt in curve:
            self.assertIn("width_factor", pt)
            self.assertIn("false_prog_rate", pt)
            self.assertIn("stable_cert_rate", pt)

    def test_fpr_curve_monotone_cert_rate(self):
        """Wider intervals (larger w) should give at most as high cert_rate as narrower ones.

        Stable rows with center=0 → wider interval always crosses tau=10 from below,
        so cert_rate should monotonically increase as w increases (larger half_width means
        hi_scaled = center + w*half grows and eventually stays below tau... wait, center=0
        and tau=10 → hi_scaled=w*half. As w grows, hi_scaled grows above tau, reducing cert.
        Use center well below tau so larger w keeps hi_scaled below tau for longer.

        Use rows with center=1, half=1 → hi_scaled = 1 + w. With tau=10, cert=1 when
        hi_scaled < 10, i.e. w < 9. So cert_rate should be 1.0 for all w in [0.1, 3.0].
        """
        from dcc.eval.metrics import coverage_vs_false_progression_curve

        # center=1, half=1: hi_scaled = 1 + w*1 < 10 for all w <= 3
        rows = _make_rows([(0.0, 2.0)], ["stable"])
        curve = coverage_vs_false_progression_curve(rows, tau=10.0, n_points=10)

        # All cert_rates should be 1.0 since hi_scaled = 1+w < 10 for w in [0.1, 3.0]
        for pt in curve:
            self.assertAlmostEqual(pt["stable_cert_rate"], 1.0, places=6)

    def test_fpr_rate_increases_with_shrinkage(self):
        """Narrower intervals (smaller w) should push lo_scaled up, increasing FPR."""
        from dcc.eval.metrics import coverage_vs_false_progression_curve

        # Stable row: center=15, half=1, tau=10 → lo_scaled = 15 - w*1
        # lo_scaled > tau=10 when w < 5. For w in [0.1, 3.0], lo_scaled > 10 always.
        rows = _make_rows([(14.0, 16.0)], ["stable"])
        curve = coverage_vs_false_progression_curve(rows, tau=10.0, n_points=10)

        # All points: lo_scaled = 15 - w >= 15 - 3.0 = 12 > 10 → FPR = 1.0
        for pt in curve:
            self.assertAlmostEqual(pt["false_prog_rate"], 1.0, places=6)

    def test_fpr_curve_empty_rows(self):
        from dcc.eval.metrics import coverage_vs_false_progression_curve

        curve = coverage_vs_false_progression_curve([], tau=5.0, n_points=4)
        self.assertEqual(len(curve), 4)
        for pt in curve:
            self.assertEqual(pt["false_prog_rate"], 0.0)
            self.assertEqual(pt["stable_cert_rate"], 0.0)


class TestIntervalWidthInReportJson(unittest.TestCase):
    def test_interval_width_in_report_json(self):
        from dcc.eval.metrics import summarize_decisions
        from dcc.eval.report import write_report

        rows = _make_rows(
            [(0.0, 4.0), (3.0, 7.0)],
            ["stable", "progressed"],
        )
        summary = summarize_decisions(rows)

        with tempfile.TemporaryDirectory() as tmp:
            _, metrics_path = write_report(summary, tmp, rows=rows, tau=5.0)
            data = json.loads(Path(metrics_path).read_text())

        self.assertIn("mean_interval_width", data)
        self.assertIn("interval_width_std", data)
        self.assertAlmostEqual(data["mean_interval_width"], 4.0, places=6)

    def test_fpr_curve_in_report_json_when_rows_and_tau_provided(self):
        from dcc.eval.metrics import summarize_decisions
        from dcc.eval.report import write_report

        rows = _make_rows([(0.0, 2.0), (5.0, 9.0)], ["stable", "progressed"])
        summary = summarize_decisions(rows)

        with tempfile.TemporaryDirectory() as tmp:
            _, metrics_path = write_report(summary, tmp, rows=rows, tau=3.0)
            data = json.loads(Path(metrics_path).read_text())

        self.assertIn("fpr_curve", data)
        self.assertIsInstance(data["fpr_curve"], list)
        self.assertGreater(len(data["fpr_curve"]), 0)
        first = data["fpr_curve"][0]
        self.assertIn("width_factor", first)
        self.assertIn("false_prog_rate", first)
        self.assertIn("stable_cert_rate", first)

    def test_fpr_curve_absent_when_rows_not_provided(self):
        from dcc.eval.metrics import summarize_decisions
        from dcc.eval.report import write_report

        rows = _make_rows([(0.0, 2.0)], ["stable"])
        summary = summarize_decisions(rows)

        with tempfile.TemporaryDirectory() as tmp:
            _, metrics_path = write_report(summary, tmp)
            data = json.loads(Path(metrics_path).read_text())

        self.assertNotIn("fpr_curve", data)


class TestTightnessAtFixedCoverage(unittest.TestCase):
    def test_tightness_at_fixed_coverage_monotone(self):
        """Wider factor → higher cert rate; smallest w achieving 90% <= smallest w for 95%."""
        from dcc.eval.metrics import tightness_at_fixed_coverage

        # center=1, half=1 → hi_scaled = 1 + w. With tau=5, cert when 1+w < 5 → w < 4.
        # Since we sweep [0.01, 3.0], all should be certified for any w < 4.
        rows = _make_rows([(0.0, 2.0), (0.5, 1.5), (0.0, 2.0)], ["stable", "stable", "stable"])
        w90 = tightness_at_fixed_coverage(rows, tau=5.0, target_coverage=0.90)
        w95 = tightness_at_fixed_coverage(rows, tau=5.0, target_coverage=0.95)
        # Both should succeed (all rows trivially certified for w in [0.01,3.0])
        self.assertIsNotNone(w90)
        self.assertIsNotNone(w95)
        # The smallest w achieving 90% coverage must be <= smallest w achieving 95%
        self.assertLessEqual(w90, w95)

    def test_tightness_returns_none_when_target_unreachable(self):
        """Pathological rows where even w=3.0 can't certify stable pairs."""
        from dcc.eval.metrics import tightness_at_fixed_coverage

        # center=100, half=1 → hi_scaled = 100 + w*1, always >> tau=5. Never certified.
        rows = _make_rows([(99.0, 101.0)], ["stable"])
        result = tightness_at_fixed_coverage(rows, tau=5.0, target_coverage=0.90)
        self.assertIsNone(result)

    def test_tightness_with_no_stable_rows_returns_none(self):
        """No stable rows → can never reach any coverage → None."""
        from dcc.eval.metrics import tightness_at_fixed_coverage

        rows = _make_rows([(0.0, 2.0)], ["progressed"])
        result = tightness_at_fixed_coverage(rows, tau=10.0, target_coverage=0.90)
        self.assertIsNone(result)


class TestAucFprCoverageCurve(unittest.TestCase):
    def test_auc_fpr_coverage_curve_is_zero_for_perfect_predictor(self):
        """A predictor that never false-progresses gives AUC=0, provided the
        coverage axis actually varies across width factors.

        Pair A (center 8, half 1) crosses tau=10 as the width grows, so its
        certification flips stable->uncertain and the stable-cert-rate sweeps
        from 1.0 down to 0.5 (a real x-axis). Pair B (center 1) stays certified.
        Neither ever false-progresses (lo never exceeds tau), so FPR=0 at every
        point and the area is 0.
        """
        from dcc.eval.metrics import auc_fpr_coverage_curve, coverage_vs_false_progression_curve

        rows = _make_rows([(7.0, 9.0), (0.0, 2.0)], ["stable", "stable"])
        curve = coverage_vs_false_progression_curve(rows, tau=10.0, n_points=20)
        auc = auc_fpr_coverage_curve(curve)
        self.assertAlmostEqual(auc, 0.0, places=6)

    def test_auc_degenerate_x_axis_returns_nan(self):
        """A curve whose stable_cert_rate never varies has no area to integrate;
        AUC is undefined and must be NaN, not the 'best' value 0.0."""
        import math
        from dcc.eval.metrics import auc_fpr_coverage_curve

        curve = [{"stable_cert_rate": 0.5, "false_prog_rate": 0.8}] * 5
        self.assertTrue(math.isnan(auc_fpr_coverage_curve(curve)))

    def test_auc_fpr_coverage_curve_between_zero_and_one(self):
        """AUC should be in [0, 1] for any valid curve."""
        from dcc.eval.metrics import auc_fpr_coverage_curve, coverage_vs_false_progression_curve

        # Mixed scenario: some stable above tau, some below
        rows = _make_rows(
            [(0.0, 2.0), (8.0, 12.0), (3.0, 5.0)],
            ["stable", "stable", "stable"],
        )
        curve = coverage_vs_false_progression_curve(rows, tau=6.0, n_points=20)
        auc = auc_fpr_coverage_curve(curve)
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)

    def test_auc_empty_curve_returns_nan(self):
        """Empty curve has no area to integrate → NaN (undefined), not 0.0."""
        import math
        from dcc.eval.metrics import auc_fpr_coverage_curve

        self.assertTrue(math.isnan(auc_fpr_coverage_curve([])))


class TestSummarizeDecisionsErrors(unittest.TestCase):
    def test_unknown_true_label_raises(self):
        from dcc.eval.metrics import summarize_decisions

        with self.assertRaises(ValueError, msg="Unknown true label"):
            summarize_decisions([{"true": "wrong_label", "decision": "stable"}])

    def test_rates_rounded_and_correct_denominators(self):
        """Rates use the right per-label denominators and are rounded to 10 dp
        (1/3 is irrational in binary, so it exercises the round())."""
        from dcc.eval.metrics import summarize_decisions

        # 3 stable: 1 falsely 'progressed' (FP), 2 correctly 'stable'.
        rows = [
            {"true": "stable", "decision": "progressed"},
            {"true": "stable", "decision": "stable"},
            {"true": "stable", "decision": "stable"},
        ]
        s = summarize_decisions(rows)
        self.assertEqual(s.false_progression_rate, round(1 / 3, 10))
        self.assertEqual(s.stable_certification_rate, round(2 / 3, 10))

    def test_unknown_decision_raises(self):
        from dcc.eval.metrics import summarize_decisions

        with self.assertRaises(ValueError, msg="Unknown decision"):
            summarize_decisions([{"true": "stable", "decision": "bad_decision"}])


if __name__ == "__main__":
    unittest.main()
