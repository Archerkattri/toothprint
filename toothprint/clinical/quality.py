"""Input quality gates — refuse unusable captures instead of certifying garbage.

A clinical system must reject a blurred radiograph or an incomplete scan rather
than emit a confident-looking verdict on it. These gates score acquisition
quality and return a hard ``usable`` flag with the reasons, so the decision layer
can abstain ("refer / recapture") on poor input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityReport:
    usable: bool
    metrics: dict
    issues: list


def assess_radiograph(gray: np.ndarray, *, min_sharpness: float = 6.0,
                      min_contrast: float = 12.0, min_side: int = 128) -> QualityReport:
    """Gate a greyscale radiograph on sharpness, contrast, and size.

    ``sharpness`` is the variance of the image gradient magnitude (blur -> low);
    ``contrast`` is the intensity standard deviation. Thresholds are conservative
    defaults to be re-tuned per site.
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("radiograph must be a 2D greyscale array")
    h, w = g.shape
    gy, gx = np.gradient(g)
    sharpness = float(np.sqrt(gx ** 2 + gy ** 2).var())
    contrast = float(g.std())
    issues = []
    if min(h, w) < min_side:
        issues.append(f"too small ({w}x{h} < {min_side})")
    if sharpness < min_sharpness:
        issues.append(f"too blurred (sharpness {sharpness:.1f} < {min_sharpness})")
    if contrast < min_contrast:
        issues.append(f"too low contrast ({contrast:.1f} < {min_contrast})")
    return QualityReport(usable=not issues,
                         metrics={"sharpness": sharpness, "contrast": contrast, "height": h, "width": w},
                         issues=issues)


def assess_scan(points: np.ndarray, *, min_points: int = 1500,
                min_extent_mm: float = 20.0) -> QualityReport:
    """Gate a 3D scan point cloud on point count and spatial extent."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("scan must be an (N, 3) array")
    n = pts.shape[0]
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0))) if n else 0.0
    issues = []
    if n < min_points:
        issues.append(f"too sparse ({n} < {min_points} points)")
    if extent < min_extent_mm:
        issues.append(f"too small ({extent:.1f}mm < {min_extent_mm}mm) — incomplete arch?")
    return QualityReport(usable=not issues,
                         metrics={"n_points": n, "extent_mm": extent}, issues=issues)
