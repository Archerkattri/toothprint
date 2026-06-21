"""Adversarial acquisition-only perturbation search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from toothprint.bench.perturb.acquisition import TransformParams, apply_acquisition_perturbation
from toothprint.bench.score.periodontal import scalar_change_score


@dataclass(frozen=True)
class ConfounderResult:
    params: TransformParams | None
    score: float


def maximize_artifact_score(
    annotation: dict,
    candidates: Iterable[TransformParams],
    tooth_id: str | None = None,
) -> ConfounderResult:
    """Find the candidate perturbation that produces the highest change score.

    A score of exactly 0.0 is a genuine result (no detectable landmark
    displacement).  It is used as-is rather than being replaced by a proxy
    value so that the confounder search reflects the true output of the
    scoring pipeline.
    """
    best: ConfounderResult | None = None
    for params in candidates:
        pair = apply_acquisition_perturbation(annotation, params)
        score = scalar_change_score(pair.baseline, pair.followup, tooth_id=tooth_id)
        result = ConfounderResult(params=params, score=score)
        if best is None or result.score > best.score:
            best = result
    if best is None:
        raise ValueError("At least one candidate perturbation is required")
    return best


def maximize_artifact_score_full(
    annotation: dict,
    rigid_candidates: Iterable[TransformParams],
    per_landmark_deltas: Iterable[float] | None = None,
    exposure_deltas: Iterable[float] | None = None,
    tooth_id: str | None = None,
    seed: int = 0,
) -> ConfounderResult:
    """Adversarial search over rigid, per-landmark, and exposure perturbations.

    Returns the candidate with the highest change score across all families.
    """
    from toothprint.bench.perturb.acquisition import apply_per_landmark_perturbation, apply_exposure_perturbation

    if per_landmark_deltas is None:
        per_landmark_deltas = [3.0, 5.0, 8.0, 12.0]
    if exposure_deltas is None:
        exposure_deltas = [0.1, 0.2, 0.3, -0.1, -0.2]

    best: ConfounderResult | None = None

    def _check(pair, params):
        nonlocal best
        score = scalar_change_score(pair.baseline, pair.followup, tooth_id=tooth_id)
        result = ConfounderResult(params=params, score=score)
        if best is None or result.score > best.score:
            best = result

    # Rigid candidates
    for params in rigid_candidates:
        pair = apply_acquisition_perturbation(annotation, params)
        _check(pair, params)

    # Per-landmark independent perturbation
    for delta in per_landmark_deltas:
        pair = apply_per_landmark_perturbation(annotation, dx_per_point=delta, dy_per_point=delta, seed=seed)
        _check(pair, TransformParams(metadata={"family": "per_landmark", "delta_px": delta}))

    # Exposure-only perturbation (should always produce score≈0)
    for exp in exposure_deltas:
        pair = apply_exposure_perturbation(annotation, exposure_delta=exp)
        _check(pair, TransformParams(exposure_delta=exp, metadata={"family": "exposure"}))

    if best is None:
        raise ValueError("At least one candidate is required")
    return best
