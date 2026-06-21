"""Evaluate conformal coverage under distribution shift.

Compares standard symmetric conformal vs weighted conformal under:
  - Perturbation-family shift (different acquisition noise std)
  - Cross-source shift (different dataset)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from toothprint.bench.certificate.conformal import ConformalInterval, AsymmetricConformalInterval
from toothprint.bench.certificate.weighted_conformal import WeightedConformalInterval, perturbation_shift_weights, cross_source_weights


@dataclass(frozen=True)
class ShiftEvalResult:
    """Coverage comparison under a specific shift."""
    shift_type: str
    standard_coverage: float      # empirical coverage with standard conformal
    weighted_coverage: float      # empirical coverage with weighted conformal
    standard_mean_width: float    # mean interval width (standard)
    weighted_mean_width: float    # mean interval width (weighted)
    n_test: int
    alpha: float


def evaluate_under_perturbation_shift(
    cal_pairs: list,
    test_pairs: list,
    alpha: float = 0.1,
    cal_noise_std: float = 3.0,
    test_noise_std: float = 10.0,
) -> ShiftEvalResult:
    """Compare standard vs weighted conformal under perturbation-family shift.

    cal_pairs: calibration pairs (built with acq_noise_std=cal_noise_std)
    test_pairs: test pairs (built with acq_noise_std=test_noise_std)
    """
    from toothprint.bench.score.periodontal import scalar_change_score

    cal_predicted = [scalar_change_score(p.baseline, p.followup) for p in cal_pairs]
    cal_observed = [p.true_change for p in cal_pairs]
    cal_residuals = [abs(p - o) for p, o in zip(cal_predicted, cal_observed)]

    test_predicted = [scalar_change_score(p.baseline, p.followup) for p in test_pairs]
    test_observed = [p.true_change for p in test_pairs]

    # Standard conformal
    std_ci = ConformalInterval.fit(cal_predicted, cal_observed, alpha=alpha)
    std_covered = [
        std_ci.predict(pred)[0] <= obs <= std_ci.predict(pred)[1]
        for pred, obs in zip(test_predicted, test_observed)
    ]
    std_widths = [std_ci.radius * 2] * len(test_pairs)

    # Weighted conformal
    cal_noise_stds = [cal_noise_std] * len(cal_pairs)
    weights = perturbation_shift_weights(cal_noise_stds, test_noise_std)
    w_ci = WeightedConformalInterval.fit(cal_residuals, weights, alpha=alpha)
    w_covered = [
        w_ci.predict(pred)[0] <= obs <= w_ci.predict(pred)[1]
        for pred, obs in zip(test_predicted, test_observed)
    ]
    w_widths = [w_ci.radius * 2] * len(test_pairs)

    n = len(test_pairs)
    return ShiftEvalResult(
        shift_type="perturbation_family",
        standard_coverage=sum(std_covered) / n if n > 0 else 0.0,
        weighted_coverage=sum(w_covered) / n if n > 0 else 0.0,
        standard_mean_width=float(np.mean(std_widths)),
        weighted_mean_width=float(np.mean(w_widths)),
        n_test=n,
        alpha=alpha,
    )


def evaluate_under_cross_source_shift(
    cal_pairs: list,
    cal_sources: list[str],
    test_pairs: list,
    test_source: str,
    alpha: float = 0.1,
    downweight_ratio: float = 0.1,
) -> ShiftEvalResult:
    """Compare standard vs weighted conformal under cross-source shift."""
    from toothprint.bench.score.periodontal import scalar_change_score

    cal_predicted = [scalar_change_score(p.baseline, p.followup) for p in cal_pairs]
    cal_observed = [p.true_change for p in cal_pairs]
    cal_residuals = [abs(p - o) for p, o in zip(cal_predicted, cal_observed)]

    test_predicted = [scalar_change_score(p.baseline, p.followup) for p in test_pairs]
    test_observed = [p.true_change for p in test_pairs]

    # Standard conformal
    std_ci = ConformalInterval.fit(cal_predicted, cal_observed, alpha=alpha)
    std_covered = [
        std_ci.predict(pred)[0] <= obs <= std_ci.predict(pred)[1]
        for pred, obs in zip(test_predicted, test_observed)
    ]

    # Weighted conformal
    weights = cross_source_weights(cal_sources, test_source, downweight_ratio)
    w_ci = WeightedConformalInterval.fit(cal_residuals, weights, alpha=alpha)
    w_covered = [
        w_ci.predict(pred)[0] <= obs <= w_ci.predict(pred)[1]
        for pred, obs in zip(test_predicted, test_observed)
    ]

    n = len(test_pairs)
    return ShiftEvalResult(
        shift_type="cross_source",
        standard_coverage=sum(std_covered) / n if n > 0 else 0.0,
        weighted_coverage=sum(w_covered) / n if n > 0 else 0.0,
        standard_mean_width=float(np.mean([std_ci.radius * 2] * n)),
        weighted_mean_width=float(np.mean([w_ci.radius * 2] * n)),
        n_test=n,
        alpha=alpha,
    )
