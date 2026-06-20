import unittest


class CertificateTests(unittest.TestCase):
    def test_deterministic_decision_thresholds_change_score(self):
        from dcc.certificate.deterministic import decide

        self.assertEqual(decide(1.2, tau=1.0), "progressed")
        self.assertEqual(decide(0.2, tau=1.0), "stable")

    def test_deterministic_decision_at_exact_threshold_is_stable(self):
        """score == tau must NOT be 'progressed' (strict >). The boundary is
        clinically load-bearing: equality is the conservative 'stable'."""
        from dcc.certificate.deterministic import decide

        self.assertEqual(decide(1.0, tau=1.0), "stable")
        self.assertEqual(decide(1.0 + 1e-9, tau=1.0), "progressed")

    def test_deterministic_decision_rejects_non_finite_score(self):
        """A NaN/inf score must raise rather than silently default to 'stable'."""
        from dcc.certificate.deterministic import decide

        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                decide(bad, tau=1.0)

    def test_interval_classifier_abstains_when_threshold_is_inside_interval(self):
        from dcc.certificate.conformal import classify_interval

        self.assertEqual(classify_interval((1.2, 2.0), tau=1.0), "progressed")
        self.assertEqual(classify_interval((0.1, 0.8), tau=1.0), "stable")
        self.assertEqual(classify_interval((0.8, 1.2), tau=1.0), "uncertain")

    def test_classify_interval_at_boundaries_abstains(self):
        """tau exactly on a bound abstains (strict < / >): the interval includes
        tau, so neither 'progressed' nor 'stable' can be certified."""
        from dcc.certificate.conformal import classify_interval

        self.assertEqual(classify_interval((1.0, 2.0), tau=1.0), "uncertain")
        self.assertEqual(classify_interval((1.0, 2.0), tau=2.0), "uncertain")

    def test_conformal_interval_uses_calibration_residual_quantile(self):
        from dcc.certificate.conformal import ConformalInterval

        model = ConformalInterval.fit(
            predicted=[1.0, 2.0, 3.0],
            observed=[1.2, 1.7, 3.4],
            alpha=0.34,
        )

        self.assertEqual(model.radius, 0.4)
        self.assertEqual(model.predict(2.5), (2.1, 2.9))

    def test_raw_mc_interval_uses_min_max_scores(self):
        from dcc.certificate.raw_mc import raw_interval

        self.assertEqual(raw_interval([0.4, 1.2, 0.8]), (0.4, 1.2))

    def test_raw_mc_interval_raises_on_empty_scores(self):
        from dcc.certificate.raw_mc import raw_interval

        with self.assertRaises(ValueError):
            raw_interval([])

    def test_oracle_interval_width_equals_twice_noise_budget(self):
        """FIX 5: oracle_interval produces [score-budget, score+budget]."""
        from dcc.certificate.oracle import oracle_interval

        lo, hi = oracle_interval(10.0, noise_budget_px=3.0)
        self.assertAlmostEqual(lo, 7.0)
        self.assertAlmostEqual(hi, 13.0)

    def test_oracle_interval_lo_clamped_at_zero(self):
        """FIX 5: lo is clamped to 0 so it never goes negative."""
        from dcc.certificate.oracle import oracle_interval

        lo, hi = oracle_interval(2.0, noise_budget_px=5.0)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 7.0)

    def test_oracle_interval_tighter_than_conformal_with_large_cal_set(self):
        """FIX 5: oracle interval is tighter than conformal when noise budget is known
        and conformal residuals are inflated by model bias.

        Conformal uses empirical |predicted - observed| residuals which include both
        acquisition noise AND model prediction error. The oracle only uses the known
        acquisition noise budget, so it is tighter when model bias is large.
        """
        from dcc.certificate.conformal import ConformalInterval
        from dcc.certificate.oracle import oracle_interval
        import random

        rng = random.Random(42)
        noise_std = 1.0
        # Oracle uses 1-sigma budget (tight, known acquisition noise)
        noise_budget_px = noise_std  # 1px half-width → 2px oracle width

        # Calibration set has large model residuals (bias of ~5px) → wide conformal interval
        n_cal = 60
        predicted = [float(i) for i in range(n_cal)]
        # Residuals inflated by a 5px constant bias (simulating model error)
        observed = [p + 5.0 + rng.gauss(0, noise_std) for p in predicted]

        conformal = ConformalInterval.fit(predicted, observed, alpha=0.1)
        score = 15.0
        c_lo, c_hi = conformal.predict(score)
        o_lo, o_hi = oracle_interval(score, noise_budget_px=noise_budget_px)

        oracle_width = o_hi - o_lo
        conformal_width = c_hi - c_lo
        self.assertLess(oracle_width, conformal_width,
                        f"oracle width {oracle_width:.3f} should be < conformal width {conformal_width:.3f}")


if __name__ == "__main__":
    unittest.main()
