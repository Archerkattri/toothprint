"""Gallery-level, selection-aware identity calibration.

Pairwise false-match calibration is not enough after a matcher searches an
entire gallery: the decision is made from the best (minimum-distance) candidate
selected from all comparisons.  This module calibrates that *whole-gallery*
selection unit on non-enrolled queries and returns an explicit accept/abstain
decision.  It also returns an identity set when multiple gallery labels pass the
same threshold, making ambiguity visible instead of silently choosing argmin.

The guarantee is finite-sample and conditional on the declared gallery size,
selector, score direction, calibration population and site.  Changing the
gallery, registration/scoring pipeline or deployment distribution requires a
new calibration or a separately justified update.  This is an identity-safety
primitive, not a clinical or forensic deployment claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


METHOD_VERSION = "gallery-max-distance-v1"


def _fingerprint(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    h = hashlib.sha256()
    h.update(b"toothprint-gallery-calibration-v1\0")
    h.update(json.dumps({"dtype": arr.dtype.str, "shape": list(arr.shape)}, sort_keys=True, separators=(",", ":")).encode("ascii"))
    h.update(b"\0")
    h.update(arr.tobytes())
    return h.hexdigest()


def _finite_distances(values: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite distances")
    return arr


def _stable_quantile_threshold(best: np.ndarray, alpha: float) -> tuple[float, int]:
    """Return the k-th lower order statistic and its conformal rank.

    The acceptance event is ``distance < threshold`` (strict inequality).  The
    strict comparison is intentional: ties at the boundary are rejected rather
    than increasing the false-accept rate beyond the finite-sample lower-tail
    order-statistic budget.
    """
    n = int(best.size)
    rank = int(math.floor((n + 1) * alpha))
    if rank < 1:
        return -math.inf, 0
    ordered = np.sort(best, kind="mergesort")
    rank = min(rank, n)
    return float(ordered[rank - 1]), rank


@dataclass(frozen=True)
class GalleryCalibration:
    """A versioned lower-tail calibration of the best impostor distance."""

    threshold_distance: float
    alpha: float
    n_calibration: int
    gallery_size: int
    selector_id: str
    site_id: str
    created_utc: str
    calibration_sha256: str
    rank: int
    calibration_best_distances: tuple[float, ...]
    method_version: str = METHOD_VERSION

    def __post_init__(self) -> None:
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if int(self.n_calibration) < 1:
            raise ValueError("n_calibration must be positive")
        if int(self.gallery_size) < 1:
            raise ValueError("gallery_size must be positive")
        if not self.site_id:
            raise ValueError("site_id must be non-empty")
        if not self.selector_id:
            raise ValueError("selector_id must be non-empty")
        if len(self.calibration_best_distances) != int(self.n_calibration):
            raise ValueError("calibration_best_distances length must equal n_calibration")
        arr = np.asarray(self.calibration_best_distances, dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError("calibration_best_distances must be finite")
        threshold = float(self.threshold_distance)
        if not (math.isfinite(threshold) or threshold == -math.inf):
            raise ValueError("threshold_distance must be finite or -inf")

    @classmethod
    def fit(
        cls,
        impostor_distances: Any,
        *,
        alpha: float = 0.01,
        site_id: str,
        created_utc: str,
        selector_id: str = "complete_gallery_argmin_distance",
        min_calibration: int = 100,
    ) -> "GalleryCalibration":
        """Fit on non-enrolled queries scored against the complete gallery.

        ``impostor_distances`` has shape ``(n_non_enrolled_queries,
        n_gallery_entries)``.  Each row is reduced to its minimum before the
        lower-tail order statistic is computed.  This is the key distinction
        from a pairwise threshold: the maximum match score / minimum distance
        over the *entire selected gallery* is the calibrated random variable.
        """
        distances = _finite_distances(impostor_distances, name="impostor_distances", ndim=2)
        n, gallery_size = distances.shape
        if n < int(min_calibration):
            raise ValueError(f"gallery calibration needs >= {min_calibration} non-enrolled queries, got {n}")
        if int(min_calibration) < 1:
            raise ValueError("min_calibration must be >= 1")
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        best = distances.min(axis=1)
        threshold, rank = _stable_quantile_threshold(best, float(alpha))
        return cls(
            threshold_distance=threshold,
            alpha=float(alpha),
            n_calibration=int(n),
            gallery_size=int(gallery_size),
            selector_id=str(selector_id),
            site_id=str(site_id),
            created_utc=str(created_utc),
            calibration_sha256=_fingerprint(distances),
            rank=rank,
            calibration_best_distances=tuple(float(value) for value in best),
        )

    @property
    def calibration_id(self) -> str:
        spec = {
            "method_version": self.method_version,
            "site_id": self.site_id,
            "created_utc": self.created_utc,
            "alpha": self.alpha,
            "n_calibration": self.n_calibration,
            "gallery_size": self.gallery_size,
            "selector_id": self.selector_id,
            "calibration_sha256": self.calibration_sha256,
            "rank": self.rank,
            "threshold_distance": self.threshold_distance if math.isfinite(self.threshold_distance) else None,
        }
        encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return f"{self.site_id}:{hashlib.sha256(encoded).hexdigest()[:16]}:{self.created_utc}"

    @property
    def empirical_false_accept_rate(self) -> float:
        if self.threshold_distance == -math.inf:
            return 0.0
        best = np.asarray(self.calibration_best_distances, dtype=float)
        return float(np.mean(best < self.threshold_distance))

    def lower_tail_p_value(self, best_distance: float) -> float:
        """Return the conservative lower-tail rank p-value for one selected row."""
        best_distance = float(best_distance)
        if not math.isfinite(best_distance):
            raise ValueError("best_distance must be finite")
        best = np.asarray(self.calibration_best_distances, dtype=float)
        return float((1 + np.count_nonzero(best <= best_distance)) / (self.n_calibration + 1))

    def to_dict(self, *, include_scores: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "method_version": self.method_version,
            "calibration_id": self.calibration_id,
            "threshold_distance": None if self.threshold_distance == -math.inf else self.threshold_distance,
            "no_accept_threshold": self.threshold_distance == -math.inf,
            "alpha": self.alpha,
            "n_calibration": self.n_calibration,
            "gallery_size": self.gallery_size,
            "selector_id": self.selector_id,
            "site_id": self.site_id,
            "created_utc": self.created_utc,
            "calibration_sha256": self.calibration_sha256,
            "rank": self.rank,
            "empirical_false_accept_rate": self.empirical_false_accept_rate,
        }
        if include_scores:
            payload["calibration_best_distances"] = list(self.calibration_best_distances)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GalleryCalibration":
        if "calibration_best_distances" not in payload:
            raise ValueError("serialized gallery calibration needs calibration_best_distances")
        threshold = payload.get("threshold_distance")
        return cls(
            threshold_distance=-math.inf if payload.get("no_accept_threshold") else float(threshold),
            alpha=float(payload["alpha"]),
            n_calibration=int(payload["n_calibration"]),
            gallery_size=int(payload["gallery_size"]),
            selector_id=str(payload["selector_id"]),
            site_id=str(payload["site_id"]),
            created_utc=str(payload["created_utc"]),
            calibration_sha256=str(payload["calibration_sha256"]),
            rank=int(payload["rank"]),
            calibration_best_distances=tuple(float(value) for value in payload["calibration_best_distances"]),
            method_version=str(payload.get("method_version", METHOD_VERSION)),
        )


@dataclass(frozen=True)
class GalleryDecision:
    """One auditable accept/abstain result from a gallery search."""

    label: str | None
    accepted: bool
    reason: str
    best_index: int | None
    best_distance: float
    threshold_distance: float
    identity_set: tuple[str, ...]
    candidate_indices: tuple[int, ...]
    lower_tail_p_value: float
    calibration_id: str
    gallery_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "accepted": self.accepted,
            "abstained": not self.accepted,
            "reason": self.reason,
            "best_index": self.best_index,
            "best_distance": self.best_distance,
            "threshold_distance": None if self.threshold_distance == -math.inf else self.threshold_distance,
            "identity_set": list(self.identity_set),
            "candidate_indices": list(self.candidate_indices),
            "lower_tail_p_value": self.lower_tail_p_value,
            "calibration_id": self.calibration_id,
            "gallery_size": self.gallery_size,
        }


def decide_gallery_identity(
    distances: Any,
    calibration: GalleryCalibration,
    *,
    gallery_labels: Sequence[str] | None = None,
    ambiguity_margin: float | None = None,
) -> GalleryDecision:
    """Select the best gallery candidate, then accept only if unambiguous.

    Distances are lower-is-better.  A strict ``best < threshold`` comparison
    implements the calibrated lower-tail false-accept contract.  If more than
    one distinct label passes the threshold, or the top-two margin is at/below
    an explicitly supplied ambiguity margin, the result abstains and returns
    the candidate identity set.
    """
    row = _finite_distances(distances, name="distances").ravel()
    if row.size != calibration.gallery_size:
        raise ValueError(
            f"gallery size mismatch: calibration expects {calibration.gallery_size}, got {row.size}"
        )
    if gallery_labels is not None:
        labels = tuple(str(label) for label in gallery_labels)
        if len(labels) != row.size:
            raise ValueError("gallery_labels must match the calibrated gallery size")
    else:
        labels = tuple(str(index) for index in range(row.size))
    if ambiguity_margin is not None:
        ambiguity_margin = float(ambiguity_margin)
        if not math.isfinite(ambiguity_margin) or ambiguity_margin < 0.0:
            raise ValueError("ambiguity_margin must be a finite non-negative value")

    best_index = int(np.argmin(row))
    best_distance = float(row[best_index])
    threshold = float(calibration.threshold_distance)
    candidate_indices = tuple(int(index) for index in np.flatnonzero(row < threshold))
    identity_set = tuple(dict.fromkeys(labels[index] for index in candidate_indices))
    p_value = calibration.lower_tail_p_value(best_distance)

    if not candidate_indices:
        reason = "best_candidate_does_not_clear_gallery_calibration"
        accepted = False
        label = None
    elif len(identity_set) != 1:
        reason = "multiple_gallery_identities_clear_threshold"
        accepted = False
        label = None
    elif ambiguity_margin is not None and row.size > 1:
        ordered = np.sort(row, kind="mergesort")
        if float(ordered[1] - ordered[0]) <= ambiguity_margin:
            reason = "top_two_gallery_candidates_are_ambiguous"
            accepted = False
            label = None
        else:
            reason = "single_gallery_identity_clears_threshold"
            accepted = True
            label = identity_set[0]
    else:
        reason = "single_gallery_identity_clears_threshold"
        accepted = True
        label = identity_set[0]
    return GalleryDecision(
        label=label,
        accepted=accepted,
        reason=reason,
        best_index=best_index,
        best_distance=best_distance,
        threshold_distance=threshold,
        identity_set=identity_set,
        candidate_indices=candidate_indices,
        lower_tail_p_value=p_value,
        calibration_id=calibration.calibration_id,
        gallery_size=calibration.gallery_size,
    )


def evaluate_gallery_calibration(
    calibration: GalleryCalibration,
    *,
    impostor_distances: Any,
    genuine_distances: Any | None = None,
    genuine_labels: Sequence[str] | None = None,
    gallery_labels: Sequence[str] | None = None,
    ambiguity_margin: float | None = None,
) -> dict[str, Any]:
    """Report held-out gallery false-accept and genuine-selection behavior."""
    impostors = _finite_distances(impostor_distances, name="impostor_distances", ndim=2)
    if impostors.shape[1] != calibration.gallery_size:
        raise ValueError("held-out impostor gallery size does not match calibration")
    impostor_decisions = [
        decide_gallery_identity(row, calibration, gallery_labels=gallery_labels, ambiguity_margin=ambiguity_margin)
        for row in impostors
    ]
    report: dict[str, Any] = {
        "format": "TOOTHPRINT_GALLERY_CALIBRATION_EVALUATION",
        "calibration_id": calibration.calibration_id,
        "gallery_size": calibration.gallery_size,
        "target_alpha": calibration.alpha,
        "n_impostor": int(len(impostor_decisions)),
        "false_accept_count": int(sum(decision.accepted for decision in impostor_decisions)),
        "false_accept_rate": float(np.mean([decision.accepted for decision in impostor_decisions])),
        "impostor_ambiguous_rate": float(np.mean([decision.reason == "multiple_gallery_identities_clear_threshold" for decision in impostor_decisions])),
        "claim_boundary": [
            "false-accept control is for the declared complete gallery and selector",
            "genuine acceptance/identification is empirical and has no false-negative guarantee",
            "gallery growth or matcher/site changes require recalibration",
            "not a clinical or forensic deployment validation",
        ],
    }
    if genuine_distances is not None:
        genuine = _finite_distances(genuine_distances, name="genuine_distances", ndim=2)
        if genuine.shape[1] != calibration.gallery_size:
            raise ValueError("held-out genuine gallery size does not match calibration")
        if genuine_labels is None or gallery_labels is None:
            raise ValueError("genuine_labels and gallery_labels are required for genuine evaluation")
        if len(genuine_labels) != genuine.shape[0]:
            raise ValueError("genuine_labels must match genuine_distances rows")
        genuine_decisions = [
            decide_gallery_identity(row, calibration, gallery_labels=gallery_labels, ambiguity_margin=ambiguity_margin)
            for row in genuine
        ]
        labels = tuple(str(label) for label in genuine_labels)
        correct = [decision.accepted and decision.label == label for decision, label in zip(genuine_decisions, labels)]
        report.update({
            "n_genuine": int(len(genuine_decisions)),
            "genuine_correct_accept_rate": float(np.mean(correct)),
            "genuine_abstain_rate": float(np.mean([not decision.accepted for decision in genuine_decisions])),
            "genuine_wrong_accept_rate": float(np.mean([
                decision.accepted and decision.label != label
                for decision, label in zip(genuine_decisions, labels)
            ])),
        })
    return report


__all__ = [
    "GalleryCalibration",
    "GalleryDecision",
    "decide_gallery_identity",
    "evaluate_gallery_calibration",
]
