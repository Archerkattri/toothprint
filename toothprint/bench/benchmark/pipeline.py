"""Gate 1/2 evaluation pipeline over synthetic pair objects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import warnings

from toothprint.bench.certificate.conformal import (
    AsymmetricConformalInterval,
    ConformalInterval,
    classify_interval,
)
from toothprint.bench.score.periodontal import scalar_change_score


@dataclass
class EvalRow:
    """Evaluation result for one pair.

    predicted_score: score computed from predicted (or noisy) landmarks — the
        real signal used for decisions.
    gt_score: score computed from ground-truth oracle landmarks (None when
        no oracle store is provided, i.e., in backward-compatible mode).
    """
    true: str
    decision: str
    score: float      # alias for predicted_score; kept for backward compat
    lo: float
    hi: float
    predicted_score: float
    gt_score: Optional[float] = None

    def __getitem__(self, key: str):
        """Allow dict-style access so metrics consumers can treat rows
        uniformly whether they are plain dicts (backward-compatible path)
        or :class:`EvalRow` objects (landmark-store path)."""
        try:
            return getattr(self, key)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise KeyError(key) from exc

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def get(self, key: str, default=None):
        """dict-style ``.get`` for uniform consumption by metrics helpers."""
        return getattr(self, key, default)


def evaluate_pairs(
    pairs: Iterable[object],
    tau: float,
    conformal: Optional[Union[ConformalInterval, AsymmetricConformalInterval]] = None,
    tooth_id: Optional[str] = None,
    landmark_store: Optional[object] = None,
) -> list:
    """Evaluate pairs and return rows.

    When ``landmark_store`` is provided, returns a list of :class:`EvalRow`
    objects where ``predicted_score`` comes from the store's predicted
    landmarks (not from the GT-based pair annotations).  A ``gt_score`` field
    is also populated from the pair's own baseline/followup for oracle
    comparison.

    When ``landmark_store`` is None (default), returns a list of plain dicts
    with the original schema ``{true, decision, score, lo, hi}`` for full
    backward compatibility.
    """
    if landmark_store is not None:
        return _evaluate_with_store(pairs, tau, conformal, tooth_id, landmark_store)

    # --- backward-compatible path: return plain dicts ---
    rows: list[dict] = []
    for pair in pairs:
        score = scalar_change_score(pair.baseline, pair.followup, tooth_id=tooth_id)
        if conformal is not None:
            interval = conformal.predict(score)
            decision = classify_interval(interval, tau=tau)
            lo, hi = interval
        else:
            lo, hi = score, score
            decision = "stable" if score < tau else "progressed"
        rows.append(
            {
                "true": pair.label,
                "decision": decision,
                "score": score,
                "lo": lo,
                "hi": hi,
            }
        )
    return rows


def _store_lookup(landmark_store: object, image_ref: Optional[str]) -> Optional[object]:
    """Look up a prediction tolerating filename-vs-stem key mismatches.

    The store is keyed by ``image_id`` (filename stem).  Callers may pass the
    full filename (with extension), so try the raw reference first and fall
    back to its stem.
    """
    if not image_ref:
        return None
    pred = landmark_store.get(image_ref)
    if pred is None:
        pred = landmark_store.get(Path(image_ref).stem)
    return pred


def _evaluate_with_store(
    pairs: Iterable[object],
    tau: float,
    conformal: Optional[Union[ConformalInterval, AsymmetricConformalInterval]],
    tooth_id: Optional[str],
    landmark_store: object,
) -> list[EvalRow]:
    """Score each pair using PREDICTED landmarks from the store.

    The GT score (oracle) is also computed from the pair's own annotations
    for research comparison.  In production, only ``predicted_score`` should
    be used for decisions.

    Pairs whose baseline or followup image_id is missing from the store are
    skipped with a warning rather than silently leaking GT data.
    """
    rows: list[EvalRow] = []
    for pair in pairs:
        # GT (oracle) score — from the pair's own baseline/followup annotations
        gt_score = scalar_change_score(pair.baseline, pair.followup, tooth_id=tooth_id)

        baseline_id = pair.baseline.get("image") if isinstance(pair.baseline, dict) else None
        followup_id = pair.followup.get("image") if isinstance(pair.followup, dict) else None

        # The store keys predictions by ``image_id`` (the filename stem, e.g.
        # "Image28"), while pair annotations carry the full filename (e.g.
        # "Image28.png").  Normalise to the stem so lookups match.
        pred_base = _store_lookup(landmark_store, baseline_id)
        pred_foll = _store_lookup(landmark_store, followup_id)

        if pred_base is not None and pred_base.is_oracle:
            raise ValueError(
                f"Oracle prediction (is_oracle=True) for image {baseline_id!r} must not be used "
                "in the production scoring path. Pass a non-oracle PredictedLandmarkStore."
            )
        if pred_foll is not None and pred_foll.is_oracle:
            raise ValueError(
                f"Oracle prediction (is_oracle=True) for image {followup_id!r} must not be used "
                "in the production scoring path. Pass a non-oracle PredictedLandmarkStore."
            )

        if pred_base is None or pred_foll is None:
            # Missing prediction — skip pair rather than leak GT
            warnings.warn(
                f"Skipping pair (label={pair.label!r}): missing prediction for "
                f"baseline_id={baseline_id!r} or followup_id={followup_id!r} in landmark_store.",
                UserWarning,
                stacklevel=2,
            )
            continue

        predicted_score = scalar_change_score(
            pred_base.annotation_dict, pred_foll.annotation_dict, tooth_id=tooth_id
        )

        if conformal is not None:
            interval = conformal.predict(predicted_score)
            decision = classify_interval(interval, tau=tau)
            lo, hi = interval
        else:
            lo, hi = predicted_score, predicted_score
            decision = "stable" if predicted_score < tau else "progressed"

        rows.append(EvalRow(
            true=pair.label,
            decision=decision,
            score=predicted_score,
            lo=lo,
            hi=hi,
            predicted_score=predicted_score,
            gt_score=gt_score,
        ))
    return rows
