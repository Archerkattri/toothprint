"""Coverage scoring utilities for DentalMapCert.

This module provides:

- A ``CoverageScore`` dataclass that all callers should use.
- A ``synthetic_coverage`` heuristic useful for testing and scaffolding.
- A ``coverage_from_point_cloud`` implementation that uses an adaptive-
  resolution voxel grid and Open3D for density estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

CoverageMethod = Literal["synthetic", "reconstruction"]

# Quality tags that each deduct 0.15 from heuristic coverage.
_PENALISED_TAGS = {"glare", "blur", "occlusion", "low_detail"}


@dataclass(frozen=True)
class CoverageScore:
    """Coverage estimate for a single surface region.

    Attributes:
        surface_region_id: Matches ``SurfaceRegion.surface_region_id``.
        coverage_fraction:  Fraction of the region surface estimated as
            covered by at least one view.  Must be in [0, 1].
        stable_pixels:      Pixel count considered stably visible
            (set to 0 for synthetic estimates).
        total_pixels:       Total pixel budget for the region
            (set to 0 for synthetic estimates).
        method:             ``"synthetic"`` for heuristic estimates;
            ``"reconstruction"`` for point-cloud-derived estimates.
    """

    surface_region_id: str
    coverage_fraction: float
    stable_pixels: int
    total_pixels: int
    method: CoverageMethod


def synthetic_coverage(
    surface_region_id: str,
    n_views: int,
    quality_tags: list[str],
) -> CoverageScore:
    """Return a heuristic CoverageScore without any real reconstruction.

    Heuristic:
      - ``base_coverage = min(0.95, n_views * 0.18)``
      - Deduct 0.15 for each tag in ``{"glare", "blur", "occlusion", "low_detail"}``
      - Clamp result to [0, 1]

    Args:
        surface_region_id: ID of the surface region being scored.
        n_views:           Number of views that cover this region.
        quality_tags:      List of quality issue tags for those views.

    Returns:
        A ``CoverageScore`` with ``method="synthetic"`` and
        ``stable_pixels``/``total_pixels`` both set to 0.
    """
    base = min(0.95, n_views * 0.18)
    penalty = sum(0.15 for tag in quality_tags if tag in _PENALISED_TAGS)
    fraction = max(0.0, min(1.0, base - penalty))
    return CoverageScore(
        surface_region_id=surface_region_id,
        coverage_fraction=fraction,
        stable_pixels=0,
        total_pixels=0,
        method="synthetic",
    )


def _adaptive_grid(n_points: int) -> int:
    """Return the voxel grid resolution appropriate for *n_points*.

    Resolution tiers:
      - ``n_points < 100``   → GRID=5  (125 voxels — avoids artefacts on
        very sparse clouds)
      - ``100 ≤ n_points ≤ 10 000`` → GRID=10 (1 000 voxels)
      - ``n_points > 10 000``   → GRID=20 (8 000 voxels — finer coverage
        estimate for dense reconstructions)
    """
    if n_points < 100:
        return 5
    if n_points <= 10_000:
        return 10
    return 20


def coverage_from_point_cloud(
    surface_region_id: str,
    points: list[tuple[float, float, float]],
    region_bbox: tuple[float, ...],
    grid: int | None = None,
) -> CoverageScore:
    """Compute coverage from a reconstructed point cloud via an adaptive voxel-grid.

    Algorithm:
      1. Filter *points* to those inside *region_bbox*.
      2. Optionally voxel-downsample with Open3D for better density estimation
         (falls back gracefully if Open3D is unavailable).
      3. Divide the bbox into a GRID×GRID×GRID voxel grid where GRID is
         chosen adaptively based on the number of input points:
         - ``< 100 pts``  → GRID=5  (125 voxels)
         - ``100–10 000`` → GRID=10 (1 000 voxels)
         - ``> 10 000``   → GRID=20 (8 000 voxels)
      4. Coverage fraction = (occupied voxels) / (total voxels), capped at 0.95.

    Args:
        surface_region_id: ID of the surface region being scored.
        points:            3-D point cloud for the region as a list of
            ``(x, y, z)`` tuples. The unit is arbitrary but MUST match
            ``region_bbox`` — coverage is a ratio normalised to the bbox, so any
            consistent unit (metres or millimetres) yields identical results.
        region_bbox:       Axis-aligned bounding box encoded as
            ``(x_min, y_min, z_min, x_max, y_max, z_max)`` in the SAME unit as
            ``points``.
        grid:              Optional fixed voxel-grid resolution.  When given,
            it overrides the adaptive choice so that coverage values are
            directly comparable across clouds of differing point counts
            (the adaptive tiers otherwise introduce a discontinuity at the
            100/10 000-point boundaries).

    Returns:
        A ``CoverageScore`` with ``method="reconstruction"``.
        ``stable_pixels`` holds the count of occupied voxels;
        ``total_pixels`` holds the total voxel count for the chosen GRID.

    Limitations:
        This is a density-aware occupancy *proxy*, not a true surface-coverage
        ratio. Each point occupies at most one voxel, so the occupied count is
        bounded by ``len(points)`` while the denominator is ``GRID**3``; for
        sparse clouds the fraction is therefore structurally capped at
        ``len(points) / GRID**3`` regardless of how much surface is actually
        covered. A true surface-coverage ratio would require a reference
        surface (occupied-data-voxels / occupied-reference-voxels), which this
        signature does not take. Consequently the fraction is only comparable
        across clouds evaluated at the *same* ``grid`` and similar point counts.
    """
    if len(region_bbox) != 6:
        raise ValueError("region_bbox must have exactly 6 elements: (x_min, y_min, z_min, x_max, y_max, z_max)")

    x_min, y_min, z_min, x_max, y_max, z_max = (float(v) for v in region_bbox)

    # Choose grid resolution: a caller-supplied fixed grid (for comparability
    # across clouds) takes precedence over the adaptive heuristic.
    if grid is not None:
        if grid < 1:
            raise ValueError("grid must be a positive integer")
        GRID = grid
    else:
        GRID = _adaptive_grid(len(points))
    TOTAL_VOXELS = GRID * GRID * GRID

    # Edge-case: degenerate bbox — return zero coverage.
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        return CoverageScore(
            surface_region_id=surface_region_id,
            coverage_fraction=0.0,
            stable_pixels=0,
            total_pixels=TOTAL_VOXELS,
            method="reconstruction",
        )

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # Convert to numpy for fast filtering.
    pts_arr = np.array([(float(x), float(y), float(z)) for x, y, z in points], dtype=np.float64)

    if pts_arr.shape[0] == 0:
        return CoverageScore(
            surface_region_id=surface_region_id,
            coverage_fraction=0.0,
            stable_pixels=0,
            total_pixels=TOTAL_VOXELS,
            method="reconstruction",
        )

    # Filter to bbox.
    mask = (
        (pts_arr[:, 0] >= x_min) & (pts_arr[:, 0] <= x_max) &
        (pts_arr[:, 1] >= y_min) & (pts_arr[:, 1] <= y_max) &
        (pts_arr[:, 2] >= z_min) & (pts_arr[:, 2] <= z_max)
    )
    pts_in = pts_arr[mask]

    if pts_in.shape[0] == 0:
        return CoverageScore(
            surface_region_id=surface_region_id,
            coverage_fraction=0.0,
            stable_pixels=0,
            total_pixels=TOTAL_VOXELS,
            method="reconstruction",
        )

    # Assign points to voxels directly on the raw in-bbox points. Voxel
    # occupancy is already idempotent (multiple points in a cell collapse to one
    # cell index), so pre-downsampling adds nothing and an Open3D voxel grid —
    # offset and sized differently from this occupancy grid — can wrongly merge
    # points ACROSS occupancy cells, understating coverage on non-cubic bboxes.
    occupied: set[tuple[int, int, int]] = set()
    for px, py, pz in pts_in:
        ix = min(GRID - 1, int((px - x_min) / x_range * GRID))
        iy = min(GRID - 1, int((py - y_min) / y_range * GRID))
        iz = min(GRID - 1, int((pz - z_min) / z_range * GRID))
        occupied.add((ix, iy, iz))

    occupied_count = len(occupied)
    fraction = min(0.95, occupied_count / TOTAL_VOXELS)

    return CoverageScore(
        surface_region_id=surface_region_id,
        coverage_fraction=fraction,
        stable_pixels=occupied_count,
        total_pixels=TOTAL_VOXELS,
        method="reconstruction",
    )
