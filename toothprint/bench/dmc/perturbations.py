"""Acquisition perturbation families for DMC benchmark evaluation.

Each perturbation takes a point cloud (Nx3 array) and returns a perturbed version.
Applied before coverage_from_point_cloud to simulate degraded captures.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class PerturbResult:
    points: np.ndarray  # (N, 3) perturbed points
    family: str
    params: dict


def pose_jitter(
    points: np.ndarray,
    *,
    rotation_deg: float = 5.0,
    translation_mm: float = 1.0,
    seed: int = 0,
) -> PerturbResult:
    """Random rotation + translation around centroid."""
    if points.shape[0] == 0:
        # Empty cloud: nothing to jitter. Return it unchanged rather than
        # computing a mean over an empty slice (RuntimeWarning + NaN centroid),
        # matching the empty-cloud handling of the other perturbations.
        return PerturbResult(
            points=points.copy(),
            family="pose_jitter",
            params={
                "rotation_deg": rotation_deg,
                "translation_mm": translation_mm,
                "seed": seed,
            },
        )
    rng = np.random.default_rng(seed)
    centroid = points.mean(axis=0)
    centered = points - centroid

    # Random unit axis
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)

    # Rodrigues rotation formula
    theta = np.deg2rad(rotation_deg)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

    rotated = centered @ R.T

    # Random translation of magnitude translation_mm
    t_dir = rng.normal(size=3)
    t_dir = t_dir / np.linalg.norm(t_dir)
    translation = translation_mm * t_dir

    perturbed = rotated + centroid + translation

    return PerturbResult(
        points=perturbed,
        family="pose_jitter",
        params={
            "rotation_deg": rotation_deg,
            "translation_mm": translation_mm,
            "seed": seed,
        },
    )


def sparse_dropout(
    points: np.ndarray,
    *,
    dropout_fraction: float = 0.3,
    seed: int = 0,
) -> PerturbResult:
    """Simulate occlusion / missing view by randomly dropping points."""
    rng = np.random.default_rng(seed)
    n = len(points)
    keep = int(n * (1.0 - dropout_fraction))
    idx = rng.choice(n, size=keep, replace=False)
    idx.sort()
    return PerturbResult(
        points=points[idx],
        family="sparse_dropout",
        params={"dropout_fraction": dropout_fraction, "seed": seed},
    )


def surface_noise(
    points: np.ndarray,
    *,
    noise_std_mm: float = 0.2,
    seed: int = 0,
) -> PerturbResult:
    """Gaussian noise on each point (simulates blur / low-resolution scanner)."""
    rng = np.random.default_rng(seed)
    noisy = points + rng.normal(0.0, noise_std_mm, points.shape)
    return PerturbResult(
        points=noisy,
        family="surface_noise",
        params={"noise_std_mm": noise_std_mm, "seed": seed},
    )


def partial_occlusion(
    points: np.ndarray,
    *,
    axis: int = 0,
    fraction: float = 0.3,
) -> PerturbResult:
    """Drop points with coordinate below a percentile threshold along axis.

    Simulates lip/cheek blocking part of the arch.
    """
    n = len(points)
    n_drop = int(n * fraction)
    order = np.argsort(points[:, axis])
    keep_idx = order[n_drop:]
    keep_idx.sort()
    return PerturbResult(
        points=points[keep_idx],
        family="partial_occlusion",
        params={"axis": axis, "fraction": fraction},
    )


def missing_view(
    points: np.ndarray,
    *,
    view_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    occlusion_fraction: float = 0.30,
) -> PerturbResult:
    """Drop points visible from a specific view direction.

    Simulates a missing capture angle (e.g., anterior close-up view absent).
    Projects each point onto the view direction; drops the top occlusion_fraction
    by projection value (those most 'in front' of the missing camera).

    Unlike sparse_dropout (which is random), missing_view removes a spatially
    coherent region — the part of the arch facing the missing camera.
    """
    if points.shape[0] == 0:
        return PerturbResult(
            points=points.copy(),
            family="missing_view",
            params={
                "view_direction": view_direction,
                "occlusion_fraction": occlusion_fraction,
            },
        )
    vd = np.array(view_direction, dtype=float)
    norm = np.linalg.norm(vd)
    if norm > 0:
        vd = vd / norm
    projections = points @ vd
    threshold = np.percentile(projections, (1.0 - occlusion_fraction) * 100)
    mask = projections < threshold
    return PerturbResult(
        points=points[mask].copy(),
        family="missing_view",
        params={
            "view_direction": list(view_direction),
            "occlusion_fraction": occlusion_fraction,
        },
    )


def apply_all(points: np.ndarray, *, seed: int = 0) -> list[PerturbResult]:
    """Apply all five families with default params. Returns list of 5 PerturbResult."""
    return [
        pose_jitter(points, seed=seed),
        sparse_dropout(points, seed=seed),
        surface_noise(points, seed=seed),
        partial_occlusion(points),
        missing_view(points),
    ]
