"""Manifest schema dataclasses for DentalMapCert."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

Jaw = Literal["upper", "lower"]
Split = Literal["train", "calibration", "test", "holdout"]
RegionType = Literal[
    "anterior_crown",
    "buccal_crown",
    "lingual_or_palatal_crown",
    "occlusal_or_incisal",
    "visible_gingival_margin",
    "unknown_visible_dental_surface",
]


@dataclass(frozen=True)
class CaseManifest:
    case_id: str
    subject_id: str
    timepoint_id: str
    jaw: Jaw
    source_dataset: str
    reference_mesh_path: str
    split: Split
    reference_label_path: str | None = None
    landmark_path: str | None = None
    license: str = "unknown"
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        _require(self.case_id, "case_id")
        _require(self.subject_id, "subject_id")
        _require(self.timepoint_id, "timepoint_id")
        _require(self.reference_mesh_path, "reference_mesh_path")


@dataclass(frozen=True)
class SurfaceRegion:
    surface_region_id: str
    case_id: str
    tooth_id_fdi: int
    region_type: RegionType
    vertex_indices_path: str | None = None
    claimable: bool = True
    claim_scope: str = "visible_crown_surface"
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        _require(self.surface_region_id, "surface_region_id")
        _require(self.case_id, "case_id")
        fdi = int(self.tooth_id_fdi)
        permanent = 11 <= fdi <= 48
        deciduous = 51 <= fdi <= 85
        if not (permanent or deciduous):
            raise ValueError(
                "tooth_id_fdi must be a valid FDI tooth id: permanent [11, 48] "
                "or deciduous [51, 85]"
            )
        if (
            "subgingival" in self.claim_scope.lower()
            or "root" in self.claim_scope.lower()
        ):
            raise ValueError(
                "DentalMapCert regions cannot claim hidden/root/subgingival anatomy"
            )


@dataclass(frozen=True)
class CaptureView:
    view_id: str
    image_path: str
    camera_path: str | None = None
    intended_region: str = "unknown"
    quality_tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CaptureManifest:
    capture_id: str
    case_id: str
    capture_type: str
    views: list[CaptureView]
    perturbations: dict[str, str] = field(default_factory=dict)
    paired_capture_id: str | None = None

    def validate(self) -> None:
        _require(self.capture_id, "capture_id")
        _require(self.case_id, "case_id")
        if not self.views:
            raise ValueError("capture manifest needs at least one view")


def write_jsonl(records: Iterable[Any], path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for record in records:
        if hasattr(record, "validate"):
            record.validate()
        lines.append(json.dumps(asdict(record), sort_keys=True))
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output


def validate_fdi_id(tooth_id: int) -> bool:
    """Return True if *tooth_id* is a valid FDI tooth identifier.

    Accepts permanent teeth (11-48) and deciduous teeth (51-85).
    Callers that need the full FDI constraint without instantiating a
    :class:`SurfaceRegion` can use this standalone function.

    Args:
        tooth_id: Integer FDI tooth number.

    Returns:
        ``True`` if *tooth_id* is within the permanent (11-48) or deciduous
        (51-85) ranges; ``False`` otherwise.
    """
    fdi = int(tooth_id)
    return (11 <= fdi <= 48) or (51 <= fdi <= 85)


def _require(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} is required")
