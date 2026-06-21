"""Acquisition-only perturbations for annotation-level benchmark scaffolding."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from toothprint.bench.geometry import translate_points


@dataclass(frozen=True)
class TransformParams:
    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    exposure_delta: float = 0.0
    metadata: dict[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True)
class PerturbedPair:
    baseline: dict
    followup: dict
    label: str
    params: TransformParams | None = None
    true_change: float = 0.0


LANDMARK_FIELDS = ("cej", "apex", "crest_line")


def apply_per_landmark_perturbation(
    annotation: dict,
    dx_per_point: float = 5.0,
    dy_per_point: float = 5.0,
    seed: int = 0,
) -> "PerturbedPair":
    """Apply independent random displacement to each landmark point.

    Simulates non-rigid acquisition artifact (e.g., per-tooth repositioning
    error, film bend). Returns PerturbedPair with label="stable", true_change=0.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    followup = deepcopy(annotation)
    for tooth in followup.get("teeth", []):
        for fld in LANDMARK_FIELDS:
            pts = tooth.get(fld)
            if not pts:
                continue
            n = len(pts)
            dx = rng.uniform(-dx_per_point, dx_per_point, n)
            dy = rng.uniform(-dy_per_point, dy_per_point, n)
            tooth[fld] = [
                [float(p[0]) + float(dx[i]), float(p[1]) + float(dy[i])]
                for i, p in enumerate(pts)
            ]
    return PerturbedPair(baseline=deepcopy(annotation), followup=followup,
                         label="stable", params=None, true_change=0.0)


def apply_exposure_perturbation(annotation: dict, exposure_delta: float = 0.2) -> "PerturbedPair":
    """Record an exposure-only perturbation (no landmark displacement).

    Exposure changes alone should not affect the landmark-based score.
    Used to verify score is exposure-invariant.
    """
    followup = deepcopy(annotation)
    followup.setdefault("metadata", {})["acquisition_perturbation"] = {
        "dx": 0.0, "dy": 0.0, "scale": 1.0, "exposure_delta": exposure_delta,
    }
    return PerturbedPair(baseline=deepcopy(annotation), followup=followup,
                         label="stable", params=None, true_change=0.0)


def apply_acquisition_perturbation(annotation: dict, params: TransformParams) -> PerturbedPair:
    followup = deepcopy(annotation)
    for tooth in followup.get("teeth", []):
        for field in LANDMARK_FIELDS:
            if field in tooth:
                tooth[field] = translate_points(tooth[field], dx=params.dx, dy=params.dy, scale=params.scale)
    followup.setdefault("metadata", {})["acquisition_perturbation"] = {
        "dx": params.dx,
        "dy": params.dy,
        "scale": params.scale,
        "exposure_delta": params.exposure_delta,
        **params.metadata,
    }
    return PerturbedPair(
        baseline=deepcopy(annotation),
        followup=followup,
        label="stable",
        params=params,
        true_change=0.0,
    )
