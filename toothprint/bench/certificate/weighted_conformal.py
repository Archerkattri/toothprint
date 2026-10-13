"""Weighted conformal prediction under covariate shift.

Based on: Tibshirani et al. (2019) "Conformal Prediction Under Covariate Shift"
https://arxiv.org/abs/1904.06019

The key idea: assign each calibration residual an importance weight for the
test distribution. The stated coverage guarantee requires valid density-ratio
weights, scaled relative to the test point, and the paper's covariate-shift
assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass(frozen=True)
class WeightedConformalInterval:
    """Weighted split-conformal interval for a fixed test-point weight.

    Calibration weights must be importance ratios normalized so the test point
    has weight one. The finite-sample guarantee additionally requires valid
    density ratios and the paper's covariate-shift assumptions.

    Attributes
    ----------
    quantile : float
        The weighted (1-alpha)-quantile of calibration residuals.
    alpha : float
        Miscoverage level.
    """

    quantile: float
    alpha: float

    @classmethod
    def fit(
        cls,
        residuals: list[float],
        weights: list[float],
        alpha: float = 0.1,
    ) -> WeightedConformalInterval:
        """Fit weighted conformal interval.

        Parameters
        ----------
        residuals : list of float
            |predicted - observed| for each calibration point.
        weights : list of float
            Calibration importance ratios divided by the ratio at the test
            point, so the implicit test-point weight is one. Must be finite and
            non-negative.
        alpha : float
            Miscoverage level in (0, 1).
        """
        if not residuals or not weights:
            raise ValueError("residuals and weights must be non-empty")
        if len(residuals) != len(weights):
            raise ValueError("residuals and weights must have the same length")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

        res = np.array([abs(float(r)) for r in residuals])
        w = np.array([float(wt) for wt in weights])
        if not np.all(np.isfinite(res)):
            raise ValueError("residuals must be finite")
        if not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite")
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")

        # Add inf residual for the test point. Tibshirani et al. 2019 assign
        # the test pseudo-point an unnormalized weight of 1, matching the same
        # The test pseudo-point has weight 1 under the covariate-shift method.
        # Exact rational comparisons avoid an epsilon that could select a finite
        # quantile before the required weighted mass has actually been reached.
        exact_weights = [Fraction.from_float(float(value)) for value in w]
        total_weight = sum(exact_weights, Fraction(1))
        target_weight = (Fraction(1) - Fraction(str(alpha))) * total_weight
        order = np.argsort(res, kind="mergesort")
        cumulative_weight = Fraction(0)
        quantile = float("inf")
        for index in order:
            cumulative_weight += exact_weights[int(index)]
            if cumulative_weight >= target_weight:
                quantile = float(res[index])
                break

        return cls(quantile=quantile, alpha=alpha)

    def predict(self, predicted_score: float) -> tuple[float, float]:
        lo = float(predicted_score) - self.quantile
        hi = float(predicted_score) + self.quantile
        return (lo, hi)

    @property
    def radius(self) -> float:
        return self.quantile


def perturbation_shift_weights(
    cal_noise_stds: list[float],
    test_noise_std: float,
    sigma: float | None = None,
) -> list[float]:
    """Estimate normalized perturbation-shift weights with kernel densities.

    The returned values are KDE-based proxies normalized so the estimated
    test-point ratio is one. Since these are estimated ratios, they support an
    empirical shift comparison, not an exact finite-sample coverage guarantee.

    Parameters
    ----------
    cal_noise_stds : list of float
        Acquisition noise std for each calibration point (in pixels).
    test_noise_std : float
        Acquisition noise std for the test distribution.
    sigma : float or None
        Kernel bandwidth for density estimation. Defaults to
        1.06 * std(cal_noise_stds) * n^(-1/5) (Silverman's rule).
    """
    cal = np.array([float(x) for x in cal_noise_stds])
    n = len(cal)
    test_noise_std = float(test_noise_std)
    if n == 0:
        raise ValueError("cal_noise_stds must be non-empty")
    if not np.all(np.isfinite(cal)) or not np.isfinite(test_noise_std):
        raise ValueError("noise standard deviations must be finite")
    if sigma is None:
        s = np.std(cal) if n > 1 else 1.0
        sigma = 1.06 * s * (n ** (-0.2)) if s > 0 else 1.0
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")

    def kde(x_eval: float, data: np.ndarray) -> float:
        return float(
            np.mean(np.exp(-0.5 * ((data - x_eval) / sigma) ** 2))
            / (sigma * np.sqrt(2 * np.pi))
        )

    test_distribution = np.array([test_noise_std])

    def raw_ratio(x_eval: float) -> float:
        return kde(x_eval, test_distribution) / (kde(x_eval, cal) + 1e-10)

    test_ratio = raw_ratio(test_noise_std)
    if not np.isfinite(test_ratio) or test_ratio <= 0.0:
        raise ValueError("test-point density ratio must be finite and positive")
    weights = [raw_ratio(x) / test_ratio for x in cal]
    if not np.all(np.isfinite(weights)):
        raise ValueError("estimated importance weights must be finite")
    return weights


def cross_source_weights(
    cal_sources: list[str],
    test_source: str,
    downweight_ratio: float = 0.1,
) -> list[float]:
    """Return heuristic relative weights for cross-source sensitivity analysis.

    Calibration points from the same source as test get weight 1.0.
    Calibration points from different sources get weight `downweight_ratio`.
    These hand-set values are not density-ratio estimates and do not by
    themselves support the weighted conformal finite-sample guarantee.

    Parameters
    ----------
    cal_sources : list of str
        Source dataset identifier for each calibration point (e.g., "denpar").
    test_source : str
        Source dataset identifier for the test distribution.
    downweight_ratio : float
        Weight assigned to out-of-source calibration points.
    """
    downweight_ratio = float(downweight_ratio)
    if not np.isfinite(downweight_ratio) or downweight_ratio < 0.0:
        raise ValueError("downweight_ratio must be finite and non-negative")
    return [1.0 if src == test_source else downweight_ratio for src in cal_sources]
