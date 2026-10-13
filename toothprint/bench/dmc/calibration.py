"""Calibration containers for visible-surface reconstruction error."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from toothprint._conformal import upper_order_index


@dataclass(frozen=True)
class ErrorCalibrator:
    """One-sided finite-sample residual calibrator; ``inf`` means abstain."""

    radius_mm: float
    alpha: float = 0.1

    @classmethod
    def fit(cls, residuals_mm: list[float], *, alpha: float = 0.1) -> ErrorCalibrator:
        if not residuals_mm:
            raise ValueError("at least one residual is required")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        residuals = sorted(float(x) for x in residuals_mm)
        if not all(isfinite(value) for value in residuals):
            raise ValueError("residuals must be finite")
        if residuals[0] < 0.0:
            raise ValueError("residuals must be non-negative")
        index = upper_order_index(len(residuals), alpha)
        radius = float("inf") if index >= len(residuals) else residuals[index]
        return cls(radius_mm=radius, alpha=alpha)

    def interval(self, estimate_mm: float) -> tuple[float, float]:
        estimate = max(0.0, float(estimate_mm))
        return (max(0.0, estimate - self.radius_mm), estimate + self.radius_mm)
