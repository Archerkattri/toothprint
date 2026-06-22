"""Calibration containers for visible-surface reconstruction error."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class ErrorCalibrator:
    radius_mm: float
    alpha: float = 0.1

    @classmethod
    def fit(cls, residuals_mm: list[float], *, alpha: float = 0.1) -> "ErrorCalibrator":
        if not residuals_mm:
            raise ValueError("at least one residual is required")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        residuals = sorted(float(x) for x in residuals_mm)
        if residuals[0] < 0.0:
            raise ValueError("residuals must be non-negative")
        index = min(
            len(residuals) - 1, max(0, ceil((len(residuals) + 1) * (1.0 - alpha)) - 1)
        )
        return cls(radius_mm=residuals[index], alpha=alpha)

    def interval(self, estimate_mm: float) -> tuple[float, float]:
        estimate = max(0.0, float(estimate_mm))
        return (max(0.0, estimate - self.radius_mm), estimate + self.radius_mm)
