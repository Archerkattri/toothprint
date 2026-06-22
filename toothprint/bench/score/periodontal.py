"""Periodontal bone-level change scores from predicted landmarks."""

from __future__ import annotations

from toothprint.bench.geometry import distance, mean_point


def tooth_bone_level(tooth: dict) -> float | None:
    """Return the CEJ-to-crest distance in pixels, or None if either is missing/empty."""
    cej = tooth.get("cej", [])
    crest = tooth.get("crest_line", [])
    if not cej or not crest:
        return None
    cej_mid = mean_point(cej)
    crest_mid = mean_point(crest)
    return round(distance(cej_mid, crest_mid), 10)


def record_change_scores(baseline: dict, followup: dict) -> dict[str, float]:
    baseline_by_tooth = _by_tooth_id(baseline)
    followup_by_tooth = _by_tooth_id(followup)
    scores = {}
    for tooth_id, base_tooth in baseline_by_tooth.items():
        if tooth_id not in followup_by_tooth:
            continue
        base_level = tooth_bone_level(base_tooth)
        follow_level = tooth_bone_level(followup_by_tooth[tooth_id])
        if base_level is None or follow_level is None:
            continue
        scores[tooth_id] = round(follow_level - base_level, 10)
    return scores


def scalar_change_score(
    baseline: dict, followup: dict, tooth_id: str | None = None
) -> float:
    scores = record_change_scores(baseline, followup)
    if tooth_id is not None:
        return scores[str(tooth_id)]
    if not scores:
        raise ValueError("No overlapping teeth to score")
    return max(scores.values(), key=abs)


def _by_tooth_id(annotation: dict) -> dict[str, dict]:
    return {str(tooth["tooth_id"]): tooth for tooth in annotation.get("teeth", [])}
