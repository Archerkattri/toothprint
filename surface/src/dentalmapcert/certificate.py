"""Visible-surface certificate decision rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Label = Literal[
    "surface stable certified",
    "surface change certified",
    "uncertain / recapture",
    "not visible / not claimable",
]

RecaptureAction = Literal[
    "need_left_buccal_view",
    "need_right_buccal_view",
    "need_anterior_close_view",
    "need_upper_occlusal_view",
    "need_lower_occlusal_view",
    "reduce_glare",
    "increase_focus_or_distance",
    "move_cheek_or_lip",
]


@dataclass(frozen=True)
class CertificateInput:
    surface_region_id: str
    capture_id_t0: str
    capture_id_t1: str
    coverage_score_t0: float
    coverage_score_t1: float
    error_interval_mm_t0: tuple[float, float]
    error_interval_mm_t1: tuple[float, float]
    delta_interval_mm: tuple[float, float]
    claimable: bool = True
    region_type: str = "unknown_visible_dental_surface"
    quality_tags_t0: tuple[str, ...] = ()
    quality_tags_t1: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _unit("coverage_score_t0", self.coverage_score_t0)
        _unit("coverage_score_t1", self.coverage_score_t1)
        _interval("error_interval_mm_t0", self.error_interval_mm_t0)
        _interval("error_interval_mm_t1", self.error_interval_mm_t1)
        _interval("delta_interval_mm", self.delta_interval_mm)


@dataclass(frozen=True)
class CertificateOutput:
    certificate_id: str
    surface_region_id: str
    capture_id_t0: str
    capture_id_t1: str
    coverage_score_t0: float
    coverage_score_t1: float
    error_interval_mm_t0: tuple[float, float]
    error_interval_mm_t1: tuple[float, float]
    delta_interval_mm: tuple[float, float]
    label: Label
    recapture_actions: list[RecaptureAction] = field(default_factory=list)
    not_claimable_reason: str | None = None
    calibration_version: str = "gate0_v0"

    def to_dict(self) -> dict:
        return asdict(self)


def decide_surface_change(
    item: CertificateInput,
    *,
    coverage_threshold: float = 0.75,
    stable_threshold_mm: float = 0.35,
    change_threshold_mm: float = 0.75,
    calibration_version: str = "gate0_v0",
) -> CertificateOutput:
    """Return the conservative visible-surface certificate label.

    The rule is deliberately simple and auditable. It certifies only when both
    timepoints are visible enough and the delta interval is entirely below the
    stable threshold or entirely above the change threshold.
    """

    if stable_threshold_mm >= change_threshold_mm:
        raise ValueError("stable_threshold_mm must be smaller than change_threshold_mm")

    actions = recapture_actions(item, coverage_threshold=coverage_threshold)
    reason = None
    if not item.claimable:
        label: Label = "not visible / not claimable"
        reason = "surface region is marked non-claimable"
        actions = []  # a non-claimable surface cannot be fixed by recapturing
    elif _hidden_region(item.region_type):
        label = "not visible / not claimable"
        reason = "region is outside visible-surface scope"
        actions = []  # a hidden region cannot be made visible by recapturing
    elif min(item.coverage_score_t0, item.coverage_score_t1) < coverage_threshold:
        label = "uncertain / recapture"
    elif item.delta_interval_mm[1] <= stable_threshold_mm:
        label = "surface stable certified"
    elif item.delta_interval_mm[0] >= change_threshold_mm:
        label = "surface change certified"
    else:
        label = "uncertain / recapture"

    if label == "uncertain / recapture" and not actions:
        actions = ["increase_focus_or_distance"]

    return CertificateOutput(
        certificate_id=f"cert_{item.surface_region_id}_{item.capture_id_t0}_{item.capture_id_t1}",
        surface_region_id=item.surface_region_id,
        capture_id_t0=item.capture_id_t0,
        capture_id_t1=item.capture_id_t1,
        coverage_score_t0=item.coverage_score_t0,
        coverage_score_t1=item.coverage_score_t1,
        error_interval_mm_t0=item.error_interval_mm_t0,
        error_interval_mm_t1=item.error_interval_mm_t1,
        delta_interval_mm=item.delta_interval_mm,
        label=label,
        recapture_actions=actions,
        not_claimable_reason=reason,
        calibration_version=calibration_version,
    )


def recapture_actions(item: CertificateInput, *, coverage_threshold: float) -> list[RecaptureAction]:
    tags = set(item.quality_tags_t0) | set(item.quality_tags_t1)
    actions: list[RecaptureAction] = []
    if min(item.coverage_score_t0, item.coverage_score_t1) < coverage_threshold:
        if "left_buccal_missing" in tags:
            actions.append("need_left_buccal_view")
        if "right_buccal_missing" in tags:
            actions.append("need_right_buccal_view")
        if "anterior_missing" in tags:
            actions.append("need_anterior_close_view")
        if "upper_occlusal_missing" in tags:
            actions.append("need_upper_occlusal_view")
        if "lower_occlusal_missing" in tags:
            actions.append("need_lower_occlusal_view")
    if "glare" in tags:
        actions.append("reduce_glare")
    if "blur" in tags or "focus" in tags:
        actions.append("increase_focus_or_distance")
    # analyze_view_quality emits the generic 'occlusion' tag (it cannot tell
    # lip from cheek); accept it alongside the specific variants so an occluded
    # view — which already penalises coverage — yields an actionable instruction.
    if "occlusion" in tags or "lip_occlusion" in tags or "cheek_occlusion" in tags:
        actions.append("move_cheek_or_lip")
    return list(dict.fromkeys(actions))


def _hidden_region(region_type: str) -> bool:
    lowered = region_type.lower()
    return "root" in lowered or "subgingival" in lowered or "hidden" in lowered


def _unit(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _interval(name: str, value: tuple[float, float]) -> None:
    if len(value) != 2:
        raise ValueError(f"{name} must have [lo, hi]")
    lo, hi = float(value[0]), float(value[1])
    if lo < 0.0 or hi < lo:
        raise ValueError(f"{name} must satisfy 0 <= lo <= hi")
