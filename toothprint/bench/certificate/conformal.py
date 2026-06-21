"""Split-conformal interval on scalar change scores.

Uses ``crepes.ConformalRegressor`` for the quantile computation, which
implements the Papadopoulos et al. ceil((n+1)(1-α)/n) formula with proper
finite-sample correction.

The public API (``ConformalInterval.fit`` / ``.predict``) is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    ) -> "ConformalInterval":
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
        try:
            from crepes import ConformalRegressor
        except ImportError as exc:
            raise ImportError(
                "crepes is required for ConformalInterval. "
                "Install with: pip install 'dental-change-certificate[conformal]'"
            ) from exc

        if len(predicted) != len(observed):
            raise ValueError("predicted and observed must have the same length")
        if not predicted:
            raise ValueError("At least one calibration point is required")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

        # Absolute residuals: ConformalRegressor expects signed (y - y_hat),
        # but for symmetric intervals with pre-computed absolute residuals we
        # pass them directly as signed residuals (they are already ≥ 0).
        abs_residuals = np.array(
            [abs(float(p) - float(o)) for p, o in zip(predicted, observed)]
        )

        cr = ConformalRegressor()
        cr.fit(abs_residuals)

        # predict_int(y_hat=0, confidence=1-alpha) → [-q, +q]; radius = q
        interval = cr.predict_int(
            y_hat=np.array([0.0]),
            confidence=1.0 - alpha,
        )
        radius = float(interval[0][1])

        # If crepes returns inf, the calibration set is too small to support the
        # requested coverage: the only interval that honors the 1-alpha guarantee
        # is unbounded. Return inf (predict -> (-inf, +inf), an abstaining
        # certificate) rather than fabricating a finite radius that silently
        # under-covers. Callers should gate on check_calibration_budget first.
        return cls(radius=round(radius, 10), alpha=alpha)

    def predict(self, predicted_score: float) -> tuple[float, float]:
        lo = round(float(predicted_score) - self.radius, 10)
        hi = round(float(predicted_score) + self.radius, 10)
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
    ) -> "AsymmetricConformalInterval":
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
        raw = int(np.ceil((n + 1) * (1.0 - alpha / 2))) - 1
        if raw > n - 1:
            q_hi = q_lo = float("inf")
        else:
            idx = max(0, raw)
            q_hi = float(upper_sorted[idx])
            q_lo = float(lower_sorted[idx])

        return cls(q_lo=round(q_lo, 10), q_hi=round(q_hi, 10), alpha=alpha)

    def predict(self, predicted_score: float) -> tuple[float, float]:
        lo = round(float(predicted_score) - self.q_lo, 10)
        hi = round(float(predicted_score) + self.q_hi, 10)
        return (lo, hi)

    @property
    def radius(self) -> float:
        """Mean half-width for API compatibility with ConformalInterval."""
        return (self.q_lo + self.q_hi) / 2.0
