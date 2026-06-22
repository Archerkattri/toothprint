"""Split-conformal certification with a finite-sample coverage guarantee.

A certificate only fires when the conformal interval around the measured change
lies entirely on one side of the decision threshold. The interval is calibrated
from held-out (measured, true) pairs, so the miscoverage — here the
false-progression rate — is bounded by ``alpha`` in finite samples, with no
distributional assumptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

STABLE = "stable"
CHANGED = "changed"
UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ConformalCertifier:
    """Asymmetric (CQR-style) conformal interval around a measured change."""

    q_lo: float
    q_hi: float
    alpha: float

    @classmethod
    def fit(cls, predicted, observed, alpha: float = 0.1) -> "ConformalCertifier":
        pred = np.asarray(predicted, dtype=float)
        obs = np.asarray(observed, dtype=float)
        if pred.shape != obs.shape:
            raise ValueError("predicted and observed must have the same shape")
        if pred.size == 0:
            raise ValueError("at least one calibration pair is required")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        upper = np.sort(obs - pred)  # how much we under-predicted
        lower = np.sort(pred - obs)  # how much we over-predicted
        n = pred.size
        # Finite-sample order statistic for two-sided 1-alpha coverage. If it
        # exceeds n-1 the (n+1)-th statistic does not exist -> +inf (abstain),
        # rather than silently under-covering.
        idx = math.ceil((n + 1) * (1.0 - alpha / 2.0)) - 1
        if idx > n - 1:
            return cls(q_lo=math.inf, q_hi=math.inf, alpha=alpha)
        return cls(q_lo=float(lower[idx]), q_hi=float(upper[idx]), alpha=alpha)

    def interval(self, measured: float) -> tuple[float, float]:
        """Conformal interval covering the true change for a measured value."""
        return (
            round(float(measured) - self.q_lo, 10),
            round(float(measured) + self.q_hi, 10),
        )

    def classify(self, measured: float, tau: float) -> str:
        """Certify ``changed`` / ``stable`` / ``uncertain`` against threshold ``tau``."""
        lo, hi = self.interval(measured)
        if tau < lo:
            return CHANGED
        if tau > hi:
            return STABLE
        return UNCERTAIN
