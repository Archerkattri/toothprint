"""Weighted conformal prediction under covariate shift.

Based on: Tibshirani et al. (2019) "Conformal Prediction Under Covariate Shift"
https://arxiv.org/abs/1904.06019

The key idea: assign weight w_i to each calibration residual based on how
likely that calibration point is under the test distribution. The coverage
guarantee becomes: P(Y in interval) >= 1-alpha, under the test distribution.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class WeightedConformalInterval:
    """Weighted split conformal interval.

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
    ) -> "WeightedConformalInterval":
        """Fit weighted conformal interval.

        Parameters
        ----------
        residuals : list of float
            |predicted - observed| for each calibration point.
        weights : list of float
            Importance weights w_i = p_test(x_i) / p_cal(x_i), unnormalized.
            Higher weight = more like the test distribution.
        alpha : float
            Miscoverage level in (0, 1).
        """
        if not residuals or not weights:
            raise ValueError("residuals and weights must be non-empty")
        if len(residuals) != len(weights):
            raise ValueError("residuals and weights must have the same length")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

        n = len(residuals)
        res = np.array([abs(float(r)) for r in residuals])
        w = np.array([float(wt) for wt in weights])
        if not np.all(np.isfinite(res)):
            raise ValueError("residuals must be finite")
        if not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite")
        w = np.maximum(w, 0.0)  # clip negatives

        # Add inf residual for the test point. Tibshirani et al. 2019 assign
        # the test pseudo-point an unnormalized weight of 1, matching the same
        # scale as calibration importance weights (w_i ∝ p_test/p_cal, test ∝ 1).
        # After normalization this gives p_{n+1} = 1/(sum_i w_i + 1), which for
        # uniform weights recovers the standard 1/(n+1) per-point probability.
        inf_weight = 1.0
        res = np.append(res, np.inf)
        w = np.append(w, inf_weight)

        # Normalize
        w_sum = w.sum()
        if w_sum <= 0.0:  # pragma: no cover
            w = np.ones(len(w)) / len(w)
        else:
            w = w / w_sum

        # Sort by residual
        order = np.argsort(res)
        res_sorted = res[order]
        w_sorted = w[order]
        cumw = np.cumsum(w_sorted)

        # Find smallest q where cumulative weight >= 1-alpha. Subtract a small
        # tolerance to absorb floating-point undershoot in np.cumsum, which can
        # otherwise push the index one past the true textbook order statistic
        # and spuriously trip the inf-abstain branch for uniform weights.
        idx = np.searchsorted(cumw, (1.0 - alpha) - 1e-9)
        if idx >= len(res_sorted) or np.isinf(res_sorted[idx]):
            # The (1-alpha) weighted quantile lands on the +inf test pseudo-point:
            # the calibration weights cannot support the requested coverage, so the
            # only valid interval is unbounded. Return inf (abstain) rather than
            # substituting the max finite residual, which silently under-covers.
            quantile = float("inf")
        else:
            quantile = float(res_sorted[idx])

        return cls(quantile=round(quantile, 10), alpha=alpha)

    def predict(self, predicted_score: float) -> tuple[float, float]:
        lo = round(float(predicted_score) - self.quantile, 10)
        hi = round(float(predicted_score) + self.quantile, 10)
        return (lo, hi)

    @property
    def radius(self) -> float:
        return self.quantile


def perturbation_shift_weights(
    cal_noise_stds: list[float],
    test_noise_std: float,
    sigma: float | None = None,
) -> list[float]:
    """Compute importance weights for perturbation-family shift.

    Models calibration and test noise as Gaussian(0, noise_std).
    Weight = p_test(noise_std) / p_cal(noise_std), estimated via
    Gaussian kernel density.

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
    if sigma is None:
        s = np.std(cal) if n > 1 else 1.0
        sigma = 1.06 * s * (n ** (-0.2)) if s > 0 else 1.0

    def kde(x_eval: float, data: np.ndarray) -> float:
        return float(np.mean(np.exp(-0.5 * ((data - x_eval) / sigma) ** 2)) / (sigma * np.sqrt(2 * np.pi)))

    weights = [kde(x, np.array([test_noise_std])) / (kde(x, cal) + 1e-10) for x in cal]
    return weights


def cross_source_weights(
    cal_sources: list[str],
    test_source: str,
    downweight_ratio: float = 0.1,
) -> list[float]:
    """Compute importance weights for cross-source shift.

    Calibration points from the same source as test get weight 1.0.
    Calibration points from different sources get weight `downweight_ratio`.

    Parameters
    ----------
    cal_sources : list of str
        Source dataset identifier for each calibration point (e.g., "denpar").
    test_source : str
        Source dataset identifier for the test distribution.
    downweight_ratio : float
        Weight assigned to out-of-source calibration points.
    """
    return [
        1.0 if src == test_source else downweight_ratio
        for src in cal_sources
    ]
