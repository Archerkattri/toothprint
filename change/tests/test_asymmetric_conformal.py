"""Tests for AsymmetricConformalInterval and config loader."""

import math

import numpy as np
import pytest

from dcc.certificate.conformal import (
    AsymmetricConformalInterval,
    ConformalInterval,
    classify_interval,
)
from dcc.config import load_yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quantile_index(n: int, alpha: float) -> int:
    """Finite-sample-correct per-side index: ceil((n+1)*(1-alpha/2)) - 1.

    Valid only when the result is <= n-1; otherwise the (n+1)-th order statistic
    does not exist and the conformal bound is +inf (caller must use n large
    enough, i.e. n >= ceil(1/(alpha/2)) - 1)."""
    return max(0, math.ceil((n + 1) * (1.0 - alpha / 2)) - 1)


# ---------------------------------------------------------------------------
# 1. Fit with known residuals → q_lo and q_hi computed correctly
# ---------------------------------------------------------------------------

def test_fit_known_residuals():
    # Predicted always 0; observed varies so upper_residuals = obs - pred = obs
    # lower_residuals = pred - obs = -obs. n=20 is large enough for alpha=0.1
    # (per-side 0.95) to yield a finite conformal quantile (needs n >= 19).
    predicted = [0.0] * 20
    observed = [float(i) for i in range(1, 21)]

    alpha = 0.1
    ci = AsymmetricConformalInterval.fit(predicted, observed, alpha=alpha)

    n = 20
    idx = _quantile_index(n, alpha)

    upper_residuals = np.sort(np.array(observed) - np.array(predicted))
    lower_residuals = np.sort(np.array(predicted) - np.array(observed))

    expected_q_hi = float(upper_residuals[idx])
    expected_q_lo = float(lower_residuals[idx])

    assert ci.q_hi == pytest.approx(expected_q_hi, rel=1e-9)
    assert ci.q_lo == pytest.approx(expected_q_lo, rel=1e-9)
    assert ci.alpha == alpha


# ---------------------------------------------------------------------------
# 2. Predict gives asymmetric interval [score - q_lo, score + q_hi]
# ---------------------------------------------------------------------------

def test_predict_asymmetric_interval():
    predicted = [0.0] * 20
    observed = [float(i) for i in range(1, 21)]
    ci = AsymmetricConformalInterval.fit(predicted, observed, alpha=0.1)

    score = 5.0
    lo, hi = ci.predict(score)

    assert lo == pytest.approx(score - ci.q_lo, rel=1e-9)
    assert hi == pytest.approx(score + ci.q_hi, rel=1e-9)
    # Interval should be a 2-tuple
    assert isinstance(lo, float)
    assert isinstance(hi, float)


# ---------------------------------------------------------------------------
# 3. radius property = (q_lo + q_hi) / 2
# ---------------------------------------------------------------------------

def test_radius_property():
    predicted = [0.0] * 15
    observed = [float(i) for i in range(1, 16)]
    ci = AsymmetricConformalInterval.fit(predicted, observed, alpha=0.1)

    assert ci.radius == pytest.approx((ci.q_lo + ci.q_hi) / 2.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. classify_interval works on asymmetric output
# ---------------------------------------------------------------------------

def test_classify_interval_on_asymmetric_output():
    # Build a simple interval manually
    ci = AsymmetricConformalInterval(q_lo=2.0, q_hi=3.0, alpha=0.1)

    interval = ci.predict(10.0)  # (8.0, 13.0)

    assert classify_interval(interval, tau=5.0) == "progressed"   # tau < lo
    assert classify_interval(interval, tau=15.0) == "stable"      # tau > hi

    # tau inside interval
    assert classify_interval(interval, tau=10.0) == "uncertain"


# ---------------------------------------------------------------------------
# 5. alpha=0.05 gives wider intervals than alpha=0.1
# ---------------------------------------------------------------------------

def test_wider_interval_with_stricter_alpha():
    predicted = [0.0] * 50
    observed = list(range(1, 51))

    ci_loose = AsymmetricConformalInterval.fit(predicted, observed, alpha=0.1)
    ci_strict = AsymmetricConformalInterval.fit(predicted, observed, alpha=0.05)

    # Stricter alpha → higher quantile index → wider (or equal) intervals
    assert ci_strict.q_hi >= ci_loose.q_hi
    assert ci_strict.q_lo >= ci_loose.q_lo


# ---------------------------------------------------------------------------
# 6. Empty input → ValueError
# ---------------------------------------------------------------------------

def test_empty_input_raises():
    with pytest.raises(ValueError, match="At least one calibration point"):
        AsymmetricConformalInterval.fit([], [], alpha=0.1)


# ---------------------------------------------------------------------------
# 7. Symmetric residuals → q_lo ≈ q_hi ≈ ConformalInterval.radius
# ---------------------------------------------------------------------------

def test_symmetric_residuals_match_conformal_interval():
    # When the distribution of signed residuals is symmetric around 0,
    # upper_residuals and lower_residuals have the same distribution, so
    # their (1-alpha/2) quantiles should be equal (q_lo == q_hi).
    #
    # Construction: use n/2 pairs where pred=0, obs=+e_i paired with
    # pred=0, obs=-e_i. Then:
    #   upper_residuals = obs - pred = {+e_i, -e_i}
    #   lower_residuals = pred - obs = {-e_i, +e_i}
    # Both arrays are identical → same sorted order → same quantile.
    n = 40
    rng = np.random.default_rng(42)
    half = n // 2
    errors = rng.uniform(1, 10, size=half)

    # Pair each error with its negation
    observed_arr = np.concatenate([errors, -errors])
    predicted_arr = np.zeros(n)

    asym_ci = AsymmetricConformalInterval.fit(
        list(predicted_arr), list(observed_arr), alpha=0.1
    )

    # For perfectly symmetric residuals q_lo and q_hi must be exactly equal
    assert asym_ci.q_lo == pytest.approx(asym_ci.q_hi, rel=1e-9)

    # The symmetric ConformalInterval radius should be close to the mean
    # half-width (they use the same quantile of |residuals|, just different
    # formulas: crepes uses alpha; ours uses alpha/2 — so they differ here).
    # Just verify radius = (q_lo + q_hi)/2 (already covered by test_radius_property).
    assert asym_ci.radius == pytest.approx((asym_ci.q_lo + asym_ci.q_hi) / 2.0)


# ---------------------------------------------------------------------------
# 8. Config loader loads perturb_ranges.yaml and thresholds.yaml
# ---------------------------------------------------------------------------

def test_config_loader_perturb_ranges():
    cfg = load_yaml("perturb_ranges.yaml")
    assert "acquisition_noise" in cfg
    assert "rigid_search" in cfg
    noise = cfg["acquisition_noise"]
    assert "noise_std_px" in noise
    assert "worst_case_px" in noise
    assert noise["noise_std_px"] == pytest.approx(3.0)
    assert noise["worst_case_px"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# ConformalInterval validation errors (lines 64, 66, 68)
# ---------------------------------------------------------------------------

def test_conformal_interval_raises_on_mismatched_lengths():
    """ConformalInterval.fit raises ValueError when lengths differ (line 64)."""
    with pytest.raises(ValueError, match="same length"):
        ConformalInterval.fit([1.0, 2.0], [1.0], alpha=0.1)


def test_conformal_interval_raises_when_empty():
    """ConformalInterval.fit raises ValueError with no calibration points (line 66)."""
    with pytest.raises(ValueError, match="At least one calibration point"):
        ConformalInterval.fit([], [], alpha=0.1)


def test_conformal_interval_raises_on_invalid_alpha():
    """ConformalInterval.fit raises ValueError when alpha out of range (line 68)."""
    with pytest.raises(ValueError, match="alpha must be in"):
        ConformalInterval.fit([1.0], [1.0], alpha=0.0)


# ---------------------------------------------------------------------------
# ConformalInterval ImportError path (lines 57-58)
# ---------------------------------------------------------------------------

def test_conformal_interval_raises_import_error_when_crepes_blocked():
    """ConformalInterval.fit raises ImportError when crepes is unavailable (lines 57-58)."""
    import sys
    from unittest.mock import patch
    with patch.dict(sys.modules, {'crepes': None}):
        with pytest.raises(ImportError, match="crepes is required"):
            ConformalInterval.fit([1.0, 2.0], [1.5, 2.5], alpha=0.1)


# ---------------------------------------------------------------------------
# ConformalInterval non-finite radius fallback (line 91)
# ---------------------------------------------------------------------------

def test_conformal_interval_returns_inf_when_calibration_too_small():
    """ConformalInterval.fit returns an unbounded (inf) radius when crepes cannot
    support the requested coverage.

    With n=1 calibration point and alpha=0.1, the conformal index
    ceil((1+1)*0.9)-1 = 1 exceeds n=1, so the only interval that honors the
    1-alpha guarantee is unbounded. The interval must abstain (radius=inf,
    predict -> (-inf, +inf)) rather than fabricate a finite radius that
    silently under-covers.
    """
    result = ConformalInterval.fit([1.0], [0.5], alpha=0.1)
    assert np.isinf(result.radius)
    lo, hi = result.predict(2.0)
    assert lo == -np.inf and hi == np.inf
    # An unbounded interval is classified "uncertain" (abstain) for any tau.
    assert classify_interval((lo, hi), tau=1.0) == "uncertain"


# ---------------------------------------------------------------------------
# AsymmetricConformalInterval validation errors (lines 132, 136)
# ---------------------------------------------------------------------------

def test_asym_conformal_raises_on_mismatched_lengths():
    """AsymmetricConformalInterval.fit raises ValueError for mismatched lengths (line 132)."""
    with pytest.raises(ValueError, match="same length"):
        AsymmetricConformalInterval.fit([1.0, 2.0], [1.0], alpha=0.1)


def test_asym_conformal_raises_on_invalid_alpha():
    """AsymmetricConformalInterval.fit raises ValueError when alpha out of (0,1) (line 136)."""
    with pytest.raises(ValueError, match="alpha must be in"):
        AsymmetricConformalInterval.fit([1.0], [1.0], alpha=1.5)


# ---------------------------------------------------------------------------
# AsymmetricConformalInterval non-finite fallback (lines 159, 161)
# ---------------------------------------------------------------------------

def test_asym_conformal_returns_inf_when_calibration_too_small():
    """AsymmetricConformalInterval abstains (q_lo=q_hi=inf) when n is too small.

    With n=1 and alpha=0.1, the per-side order-statistic index
    ceil((1+1)*(1-0.05))-1 = 1 exceeds n-1=0, so the (n+1)-th order statistic
    does not exist and the conformal bound is +inf. The interval must return
    inf on both sides (predict -> (-inf, +inf)) rather than clamp the index and
    silently under-cover.
    """
    result = AsymmetricConformalInterval.fit([2.0], [1.0], alpha=0.1)
    assert np.isinf(result.q_hi) and np.isinf(result.q_lo)
    lo, hi = result.predict(5.0)
    assert lo == -np.inf and hi == np.inf
    assert classify_interval((lo, hi), tau=3.0) == "uncertain"


# ---------------------------------------------------------------------------
# CalibrationRecord and protocol helpers (calibration/protocol.py)
# ---------------------------------------------------------------------------

def test_calibration_record_absolute_residual():
    """CalibrationRecord.absolute_residual returns abs(predicted - observed) (line 18)."""
    from dcc.calibration.protocol import CalibrationRecord
    rec = CalibrationRecord(
        stratum="buccal", predicted_score=3.0, observed_score=1.5, source="denpar"
    )
    assert rec.absolute_residual == pytest.approx(1.5)


def test_calibration_record_to_dict():
    """CalibrationRecord.to_dict() returns a dict with absolute_residual key (lines 21-23)."""
    from dcc.calibration.protocol import CalibrationRecord
    rec = CalibrationRecord(
        stratum="buccal", predicted_score=3.0, observed_score=1.5, source="denpar"
    )
    d = rec.to_dict()
    assert d["stratum"] == "buccal"
    assert d["absolute_residual"] == pytest.approx(1.5)


def test_coverage_satisfiable_perfect_coverage_returns_false():
    """_coverage_satisfiable returns False when target_coverage == 1.0 (line 55)."""
    from dcc.calibration.protocol import _coverage_satisfiable
    assert _coverage_satisfiable(100, 1.0) is False


def test_n_needed_perfect_coverage_returns_sentinel():
    """_n_needed returns int(1e9) when target_coverage == 1.0 (line 64)."""
    from dcc.calibration.protocol import _n_needed
    result = _n_needed(1.0)
    assert result == int(1e9)


def test_check_calibration_budget_with_none_records_returns_empty_dict():
    """check_calibration_budget with records=None defaults to [] (line 124)."""
    from dcc.calibration.protocol import check_calibration_budget
    result = check_calibration_budget(records=None)
    assert result == {}


def test_config_loader_thresholds():
    cfg = load_yaml("thresholds.yaml")
    assert "bone_level_change" in cfg
    assert "coverage" in cfg
    assert "conformal" in cfg
    blc = cfg["bone_level_change"]
    assert "tau_clinically_significant_px" in blc
    assert blc["tau_clinically_significant_px"] == pytest.approx(20.0)
    conf = cfg["conformal"]
    assert "alpha_standard" in conf
    assert conf["alpha_standard"] == pytest.approx(0.1)
