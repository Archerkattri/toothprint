"""Split-conformal intervals on scalar change scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toothprint._conformal import upper_order_index


def classify_interval(interval: tuple[float, float], tau: float) -> str:
    lo, hi = interval
    if tau < lo:
        return "progressed"
    if tau > hi:
        return "stable"
    return "uncertain"


@dataclass(frozen=True)
class ConformalInterval:
    radius: float
    alpha: float

    @classmethod
    def fit(
        cls,
        predicted: list[float],
        observed: list[float],
        alpha: float = 0.1,
    ) -> ConformalInterval:
        """Fit a split-conformal interval from calibration residuals.

        Parameters
        ----------
        predicted:
            Model-predicted change scores on the calibration split.
        observed:
            Ground-truth change scores on the calibration split.
        alpha:
            Miscoverage level.  Coverage guarantee is 1 - alpha.

        Returns
        -------
        ConformalInterval
            Fitted interval with ``radius`` = the (1-alpha) quantile of
            |predicted - observed| using the finite-sample-correct formula.
        """
        if len(predicted) != len(observed):
            raise ValueError("predicted and observed must have the same length")
        if not predicted:
            raise ValueError("At least one calibration point is required")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

        abs_residuals = np.array(
            [abs(float(p) - float(o)) for p, o in zip(predicted, observed)]
        )
        if not np.all(np.isfinite(abs_residuals)):
            raise ValueError("predicted and observed scores must be finite")
        index = upper_order_index(len(abs_residuals), alpha)
        if index >= len(abs_residuals):
            # The required order statistic does not exist. An unbounded interval
            # preserves finite-sample coverage instead of silently under-covering.
            radius = float("inf")
        else:
            radius = float(np.sort(abs_residuals)[index])
        return cls(radius=radius, alpha=alpha)

    def predict(self, predicted_score: float) -> tuple[float, float]:
        lo = float(predicted_score) - self.radius
        hi = float(predicted_score) + self.radius
        return (lo, hi)


@dataclass(frozen=True)
class AsymmetricConformalInterval:
    q_lo: float  # amount to subtract from score for lower bound
    q_hi: float  # amount to add to score for upper bound
    alpha: float

    @classmethod
    def fit(
        cls,
        predicted: list[float],
        observed: list[float],
        alpha: float = 0.1,
    ) -> AsymmetricConformalInterval:
        """Fit a CQR-style asymmetric conformal interval from calibration residuals.

        Parameters
        ----------
        predicted:
            Model-predicted change scores on the calibration split.
        observed:
            Ground-truth change scores on the calibration split.
        alpha:
            Miscoverage level.  Coverage guarantee is 1 - alpha.

        Returns
        -------
        AsymmetricConformalInterval
            Fitted interval with separate ``q_lo`` and ``q_hi`` quantiles
            computed from signed upper and lower residuals respectively.
        """
        if len(predicted) != len(observed):
            raise ValueError("predicted and observed must have the same length")
        if not predicted:
            raise ValueError("At least one calibration point is required")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

        pred = np.array([float(p) for p in predicted])
        obs = np.array([float(o) for o in observed])
        if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(obs)):
            raise ValueError("predicted and observed scores must be finite")

        # Upper residuals: how much we under-predicted (inflate hi bound)
        upper_residuals = obs - pred
        # Lower residuals: how much we over-predicted (deflate lo bound)
        lower_residuals = pred - obs

        n = len(pred)
        upper_sorted = np.sort(upper_residuals)
        lower_sorted = np.sort(lower_residuals)

        # Conformal order-statistic index. If it exceeds n-1 the (n+1)-th order
        # statistic does not exist, so the finite-sample conformal bound is +inf:
        # n is too small to guarantee 1-alpha coverage on each side. Surface that
        # as +inf (an abstaining, unbounded interval) rather than clamping the
        # index, which would silently under-cover and mask the broken guarantee.
        index = upper_order_index(n, alpha, tails=2)
        if index >= n:
            q_hi = q_lo = float("inf")
        else:
            q_hi = float(upper_sorted[index])
            q_lo = float(lower_sorted[index])

        return cls(q_lo=q_lo, q_hi=q_hi, alpha=alpha)

    def predict(self, predicted_score: float) -> tuple[float, float]:
        lo = float(predicted_score) - self.q_lo
        hi = float(predicted_score) + self.q_hi
        return (lo, hi)

    @property
    def radius(self) -> float:
        """Mean half-width for API compatibility with ConformalInterval."""
        return (self.q_lo + self.q_hi) / 2.0
