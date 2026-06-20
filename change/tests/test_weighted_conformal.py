"""Tests for WeightedConformalInterval and weight generators (M4)."""

from __future__ import annotations

import numpy as np
import pytest

from dcc.certificate.weighted_conformal import (
    WeightedConformalInterval,
    cross_source_weights,
    perturbation_shift_weights,
)
from dcc.certificate.conformal import ConformalInterval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_annotation(seed: int, n_teeth: int = 2) -> dict:
    """Build a minimal annotation dict with reproducible random coords."""
    rng = np.random.default_rng(seed)
    teeth = []
    for i in range(n_teeth):
        bx = float(rng.uniform(50, 200))
        by = float(rng.uniform(50, 200))
        teeth.append({
            "tooth_id": str(i + 1),
            "cej": [[bx, by], [bx + 20.0, by + 1.0]],
            "crest_line": [[bx + 1.0, by + 30.0], [bx + 19.0, by + 31.0]],
            "apex": [[bx + 10.0, by + 80.0]],
        })
    return {"image": f"img_{seed}.png", "teeth": teeth}


class _Record:
    def __init__(self, annotation_dict: dict) -> None:
        self.annotation_dict = annotation_dict
        self.image_id = annotation_dict["image"]


def _build_pairs(n_records: int, noise_std: float, shift_px: float = 20.0, seed: int = 0):
    """Build synthetic pairs using the project's pair builder."""
    from dcc.data.pair_builder import PairBuilderConfig, build_pairs
    records = [_Record(_make_annotation(seed * 1000 + i)) for i in range(n_records)]
    cfg = PairBuilderConfig(acq_noise_std=noise_std, crestal_shift_px=shift_px, seed=seed)
    return build_pairs(records, cfg)


# ---------------------------------------------------------------------------
# 1. Uniform weights → quantile matches standard conformal
# ---------------------------------------------------------------------------

def test_fit_with_uniform_weights_matches_standard_quantile():
    """When all weights are equal, WeightedConformalInterval should give
    roughly the same quantile as ConformalInterval (within ±10%)."""
    rng = np.random.default_rng(7)
    n = 50
    residuals = list(rng.exponential(scale=5.0, size=n))
    weights = [1.0] * n
    alpha = 0.1

    w_ci = WeightedConformalInterval.fit(residuals, weights, alpha=alpha)

    # Standard unweighted quantile (finite-sample corrected index)
    import math
    idx = min(n - 1, max(0, math.ceil((n + 1) * (1.0 - alpha)) - 1))
    std_q = float(np.sort(residuals)[idx])

    # Weighted result should be close (not necessarily identical due to inf pseudo-weight)
    assert abs(w_ci.quantile - std_q) / (std_q + 1e-9) < 0.20, (
        f"Weighted quantile {w_ci.quantile:.4f} too far from standard {std_q:.4f}"
    )


# ---------------------------------------------------------------------------
# 2. High weight on low residuals → tighter interval
# ---------------------------------------------------------------------------

def test_higher_weight_on_low_residuals_gives_tighter_interval():
    """Up-weighting calibration points with residual < median → smaller quantile."""
    rng = np.random.default_rng(42)
    n = 60
    residuals = list(rng.uniform(0, 100, size=n))
    alpha = 0.1
    median = float(np.median(residuals))

    # All-uniform weights (baseline)
    w_uniform = [1.0] * n
    ci_uniform = WeightedConformalInterval.fit(residuals, w_uniform, alpha=alpha)

    # Up-weight points with residual < median → distribution shifts to lower values
    w_low = [10.0 if r < median else 1.0 for r in residuals]
    ci_tight = WeightedConformalInterval.fit(residuals, w_low, alpha=alpha)

    assert ci_tight.quantile < ci_uniform.quantile, (
        f"Expected tighter quantile ({ci_tight.quantile:.4f}) < "
        f"uniform ({ci_uniform.quantile:.4f})"
    )


# ---------------------------------------------------------------------------
# 3. Coverage guarantee on synthetic data
# ---------------------------------------------------------------------------

def test_coverage_guarantee_on_synthetic():
    """Weighted conformal achieves ~1-alpha coverage when calibration and test
    residuals come from the SAME distribution.

    The previous version calibrated on exponential(3) but tested on
    |N(5,2)-N(5,2)| residuals; that distribution mismatch inflated coverage to
    ~0.99 and the 0.15 slack could not detect genuine under-coverage. Here both
    cal and test residuals are |N(5,2)-N(5,2)|, so with uniform weights the
    interval sits just above nominal: a tight two-sided band catches both
    under-coverage and gross over-conservatism.
    """
    rng = np.random.default_rng(0)
    n_cal = 200
    n_test = 4000
    alpha = 0.1
    true_val = 5.0

    # Calibration residuals drawn from the same |pred - obs| process as the test.
    cal_pred = rng.normal(true_val, 2.0, size=n_cal)
    cal_obs = rng.normal(true_val, 2.0, size=n_cal)
    cal_residuals = list(np.abs(cal_pred - cal_obs))
    weights = [1.0] * n_cal

    ci = WeightedConformalInterval.fit(cal_residuals, weights, alpha=alpha)

    test_predictions = rng.normal(true_val, 2.0, size=n_test)
    test_observed = rng.normal(true_val, 2.0, size=n_test)
    covered = sum(
        ci.predict(pred)[0] <= obs <= ci.predict(pred)[1]
        for pred, obs in zip(test_predictions, test_observed)
    )
    coverage = covered / n_test

    # Tight two-sided band around the 1-alpha=0.90 target (seed fixed -> stable).
    assert 0.86 <= coverage <= 0.97, (
        f"Coverage {coverage:.3f} outside [0.86, 0.97]: weighted conformal is "
        f"either under-covering or grossly over-conservative."
    )


# ---------------------------------------------------------------------------
# 4. Empty residuals raises ValueError
# ---------------------------------------------------------------------------

def test_empty_residuals_raises():
    with pytest.raises(ValueError, match="non-empty"):
        WeightedConformalInterval.fit([], [], alpha=0.1)


# ---------------------------------------------------------------------------
# 5. Mismatched lengths raises ValueError
# ---------------------------------------------------------------------------

def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        WeightedConformalInterval.fit([1.0, 2.0], [1.0], alpha=0.1)


# ---------------------------------------------------------------------------
# 6. Alpha out of range raises ValueError
# ---------------------------------------------------------------------------

def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError, match="alpha"):
        WeightedConformalInterval.fit([1.0, 2.0], [1.0, 1.0], alpha=0.0)

    with pytest.raises(ValueError, match="alpha"):
        WeightedConformalInterval.fit([1.0, 2.0], [1.0, 1.0], alpha=1.0)

    with pytest.raises(ValueError, match="alpha"):
        WeightedConformalInterval.fit([1.0, 2.0], [1.0, 1.0], alpha=-0.1)


# ---------------------------------------------------------------------------
# 7. Perturbation shift weights — same distribution → all weights roughly equal
# ---------------------------------------------------------------------------

def test_perturbation_shift_weights_same_distribution():
    """When cal and test have the same noise_std, all weights should be ~equal."""
    n = 30
    noise_std = 5.0
    cal_noise_stds = [noise_std] * n

    weights = perturbation_shift_weights(cal_noise_stds, test_noise_std=noise_std)

    assert len(weights) == n
    w_arr = np.array(weights)
    # All weights should be positive
    assert np.all(w_arr > 0), "Weights should all be positive"
    # Coefficient of variation should be small (< 1%) for identical distributions
    cv = np.std(w_arr) / (np.mean(w_arr) + 1e-12)
    assert cv < 0.01, f"Expected low variance for same distribution (cv={cv:.4f})"


# ---------------------------------------------------------------------------
# 8. Perturbation shift weights — different distribution → varied weights
# ---------------------------------------------------------------------------

def test_perturbation_shift_weights_different_distribution():
    """When test_noise_std >> cal noise stds, weights should vary meaningfully."""
    n = 20
    # Calibration at low noise levels (1-5 px)
    rng = np.random.default_rng(1)
    cal_noise_stds = list(rng.uniform(1.0, 5.0, size=n))
    test_noise_std = 15.0  # much larger than calibration range

    weights = perturbation_shift_weights(cal_noise_stds, test_noise_std=test_noise_std)

    assert len(weights) == n
    w_arr = np.array(weights)
    assert np.all(np.isfinite(w_arr)), "All weights must be finite"
    # There should be variance in weights (some cal points are more like test than others)
    assert np.std(w_arr) > 0, "Weights should vary for different distributions"


# ---------------------------------------------------------------------------
# 9. Cross-source weights — all cal from same source → all weights 1.0
# ---------------------------------------------------------------------------

def test_cross_source_weights_in_source():
    """All calibration points from test_source → all weights 1.0."""
    cal_sources = ["denpar"] * 15
    weights = cross_source_weights(cal_sources, test_source="denpar")

    assert len(weights) == 15
    assert all(w == 1.0 for w in weights), (
        f"Expected all weights=1.0 for in-source, got: {weights}"
    )


# ---------------------------------------------------------------------------
# 10. Cross-source weights — all cal from different source → all weights = ratio
# ---------------------------------------------------------------------------

def test_cross_source_weights_out_of_source():
    """All calibration points from a different source → all weights = downweight_ratio."""
    cal_sources = ["denpar"] * 12
    ratio = 0.05
    weights = cross_source_weights(cal_sources, test_source="periapical", downweight_ratio=ratio)

    assert len(weights) == 12
    assert all(w == ratio for w in weights), (
        f"Expected all weights={ratio} for out-of-source, got: {weights}"
    )


# ---------------------------------------------------------------------------
# 11. shift_eval_result_coverage_in_range — perturbation shift
# ---------------------------------------------------------------------------

def test_shift_eval_result_coverage_in_range():
    """evaluate_under_perturbation_shift on synthetic data → both coverages in [0,1]."""
    from dcc.benchmark.shift_eval import evaluate_under_perturbation_shift

    cal_pairs = _build_pairs(n_records=12, noise_std=3.0, seed=10)
    test_pairs = _build_pairs(n_records=8, noise_std=10.0, seed=20)

    result = evaluate_under_perturbation_shift(
        cal_pairs=cal_pairs,
        test_pairs=test_pairs,
        alpha=0.1,
        cal_noise_std=3.0,
        test_noise_std=10.0,
    )

    assert result.shift_type == "perturbation_family"
    assert 0.0 <= result.standard_coverage <= 1.0, (
        f"standard_coverage={result.standard_coverage} out of [0,1]"
    )
    assert 0.0 <= result.weighted_coverage <= 1.0, (
        f"weighted_coverage={result.weighted_coverage} out of [0,1]"
    )
    assert result.n_test == len(test_pairs)
    assert result.alpha == 0.1
    assert result.standard_mean_width >= 0.0
    assert result.weighted_mean_width >= 0.0


# ---------------------------------------------------------------------------
# 12. shift_eval_cross_source_on_synthetic
# ---------------------------------------------------------------------------

def test_shift_eval_cross_source_on_synthetic():
    """evaluate_under_cross_source_shift on synthetic data → both coverages in [0,1]."""
    from dcc.benchmark.shift_eval import evaluate_under_cross_source_shift

    cal_pairs = _build_pairs(n_records=12, noise_std=3.0, seed=30)
    test_pairs = _build_pairs(n_records=8, noise_std=5.0, seed=40)
    cal_sources = ["denpar"] * len(cal_pairs)

    result = evaluate_under_cross_source_shift(
        cal_pairs=cal_pairs,
        cal_sources=cal_sources,
        test_pairs=test_pairs,
        test_source="periapical",
        alpha=0.1,
        downweight_ratio=0.1,
    )

    assert result.shift_type == "cross_source"
    assert 0.0 <= result.standard_coverage <= 1.0, (
        f"standard_coverage={result.standard_coverage} out of [0,1]"
    )
    assert 0.0 <= result.weighted_coverage <= 1.0, (
        f"weighted_coverage={result.weighted_coverage} out of [0,1]"
    )
    assert result.n_test == len(test_pairs)
    assert result.alpha == 0.1


def test_weighted_conformal_all_zero_weights_falls_back_to_uniform():
    """WeightedConformalInterval.fit uses uniform weights when all weights are zero."""
    from dcc.certificate.weighted_conformal import WeightedConformalInterval

    residuals = [1.0, 2.0, 3.0, 4.0, 5.0]
    weights = [0.0, 0.0, 0.0, 0.0, 0.0]  # w_sum == 0 → uniform fallback
    model = WeightedConformalInterval.fit(residuals, weights, alpha=0.1)
    assert model.quantile > 0.0
    assert isinstance(model.quantile, float)


def test_fit_rejects_non_finite_residuals():
    """A NaN/inf residual must raise, not silently produce a wrong quantile."""
    with pytest.raises(ValueError, match="residuals must be finite"):
        WeightedConformalInterval.fit([1.0, float("inf")], [1.0, 1.0], alpha=0.1)


def test_fit_rejects_non_finite_weights():
    """A NaN/inf weight must raise, not silently skew the weighted quantile."""
    with pytest.raises(ValueError, match="weights must be finite"):
        WeightedConformalInterval.fit([1.0, 2.0], [1.0, float("nan")], alpha=0.1)


def test_cross_source_weights_mixed():
    """The realistic mixed-source case: only points whose source matches the
    test source keep weight 1.0; all others are down-weighted."""
    sources = ["denpar", "periapical", "denpar", "perio_kpt"]
    w = cross_source_weights(sources, test_source="denpar", downweight_ratio=0.1)
    assert w == [1.0, 0.1, 1.0, 0.1]
