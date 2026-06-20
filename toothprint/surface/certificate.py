"""Certified 3D surface-change decision.

A measured surface displacement is certified ``changed`` only if its conformal
interval lies entirely above the clinically-meaningful threshold, ``stable`` only
if it lies entirely below the stable threshold, and ``uncertain`` otherwise — so
reconstruction noise is never mistaken for a real surface change.
"""
from __future__ import annotations

from dataclasses import dataclass

from toothprint.change.conformal import CHANGED, STABLE, UNCERTAIN, ConformalCertifier


@dataclass(frozen=True)
class SurfaceCertificate:
    measured_mm: float
    interval_mm: tuple
    label: str


def certify_surface_change(measured_mm: float, certifier: ConformalCertifier, *,
                           stable_threshold_mm: float = 0.35,
                           change_threshold_mm: float = 0.75) -> SurfaceCertificate:
    """Certify a measured surface displacement against the change/stable thresholds."""
    if stable_threshold_mm >= change_threshold_mm:
        raise ValueError("stable_threshold_mm must be < change_threshold_mm")
    lo, hi = certifier.interval(measured_mm)
    lo = max(0.0, lo)
    if lo >= change_threshold_mm:
        label = CHANGED
    elif hi <= stable_threshold_mm:
        label = STABLE
    else:
        label = UNCERTAIN
    return SurfaceCertificate(measured_mm=float(measured_mm), interval_mm=(lo, hi), label=label)
