"""Local DenPAR-style dataset adapter.

The adapter intentionally does not download data. It validates and indexes a local
DenPAR-like layout so Gate 0 can remain lightweight and reproducible.

Expected JSON schema for each annotation file
---------------------------------------------
Each annotation file must contain a JSON object with the following structure::

    {
        "image": "case001.png",          // filename of the corresponding image
        "teeth": [                       // list of tooth annotations
            {
                "tooth_id": "36",        // unique tooth identifier (string or int)
                "cej": [                 // cemento-enamel junction points [[x, y], ...]
                    [10.0, 20.0],
                    [30.0, 20.5]
                ],
                "crest_line": [          // crestal bone points [[x, y], ...]
                    [11.0, 35.0],
                    [29.0, 35.5]
                ],
                "apex": [               // root apex point(s) [[x, y], ...]
                    [20.0, 80.0]
                ]
            }
        ]
    }

All coordinate values are in pixels.  The ``apex`` and ``crest_line`` fields
are optional per tooth but required for bone-level scoring.  Additional fields
(e.g. ``mask``) are allowed and are passed through transparently.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class DenparRecord:
    image_id: str
    image_path: Path
    annotation_path: Path
    annotation: dict


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL_KEYS = {"image", "teeth"}
_REQUIRED_TOOTH_KEYS = {"tooth_id"}


def validate_record(record: dict) -> list[str]:
    """Validate a DenPAR-format annotation dict.

    Parameters
    ----------
    record:
        A parsed JSON annotation dict (not a ``DenparRecord`` dataclass).

    Returns
    -------
    list[str]
        A list of human-readable validation error messages.  An empty list
        means the record is valid.

    Examples
    --------
    >>> validate_record({"image": "x.png", "teeth": []})
    []
    >>> validate_record({"teeth": []})
    ["Missing required top-level key: 'image'"]
    """
    errors: list[str] = []

    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in record:
            errors.append(f"Missing required top-level key: {key!r}")

    teeth = record.get("teeth")
    if teeth is not None:
        if not isinstance(teeth, list):
            errors.append(f"'teeth' must be a list, got {type(teeth).__name__!r}")
        else:
            for i, tooth in enumerate(teeth):
                if not isinstance(tooth, dict):
                    errors.append(
                        f"teeth[{i}] must be a dict, got {type(tooth).__name__!r}"
                    )
                    continue
                for key in _REQUIRED_TOOTH_KEYS:
                    if key not in tooth:
                        errors.append(f"teeth[{i}] missing required key: {key!r}")

    return errors


class DenparAdapter:
    """Read a local DenPAR-style folder with images and JSON annotations."""

    def __init__(
        self,
        root: Path | str,
        images_dir: str = "images",
        annotations_dir: str = "annotations",
    ) -> None:
        self.root = Path(root)
        self.images_dir = self.root / images_dir
        self.annotations_dir = self.root / annotations_dir

    def iter_records(self) -> Iterator[DenparRecord]:
        self._require_dir(self.images_dir, "images")
        self._require_dir(self.annotations_dir, "annotations")

        for annotation_path in sorted(self.annotations_dir.glob("*.json")):
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            image_name = annotation.get("image")
            if not image_name:
                raise ValueError(f"Missing image field in {annotation_path}")

            image_path = self.images_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image referenced by annotation does not exist: {image_path}"
                )

            yield DenparRecord(
                image_id=image_path.stem,
                image_path=image_path,
                annotation_path=annotation_path,
                annotation=annotation,
            )

    def validate_all(self) -> dict[str, list[str]]:
        """Validate all annotation records in the dataset.

        Returns
        -------
        dict[str, list[str]]
            A mapping from annotation filename (stem) to a list of validation
            error messages.  Records with an empty list are valid.
        """
        self._require_dir(self.annotations_dir, "annotations")

        results: dict[str, list[str]] = {}
        for annotation_path in sorted(self.annotations_dir.glob("*.json")):
            try:
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                results[annotation_path.stem] = [f"JSON parse error: {exc}"]
                continue
            results[annotation_path.stem] = validate_record(annotation)
        return results

    @staticmethod
    def _require_dir(path: Path, label: str) -> None:
        if not path.is_dir():
            raise FileNotFoundError(
                f"Missing required DenPAR {label} directory: {path}"
            )


# ---------------------------------------------------------------------------
# Real DenPAR dataset adapter
# ---------------------------------------------------------------------------


def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    """Squared Euclidean distance between two 2-D points."""
    return (ax - bx) ** 2 + (ay - by) ** 2


def _build_annotation_dict(kp_data: dict, bl_data: dict) -> dict:
    """Convert raw Key-Points and Bone-Level dicts into the pipeline annotation format.

    Parameters
    ----------
    kp_data:
        Parsed JSON from a ``Key Points Annotations`` file.
    bl_data:
        Parsed JSON from a ``Bone Level Annotations`` file.

    Returns
    -------
    dict
        Annotation dict with keys ``"image"`` and ``"teeth"``.
    """
    image_id: str = kp_data["Image_id"]
    bboxes: list[list[float]] = kp_data.get("bboxes", [])
    cej_points: list[list[float]] = kp_data.get("CEJ_Points", [])
    apex_points: list[list[float]] = kp_data.get("Apex_Points", [])
    bone_lines: list[list[list[float]]] = bl_data.get("Bone_Lines", [])

    # Sort bboxes by x-center for stable tooth_id numbering.
    bboxes_sorted = sorted(bboxes, key=lambda b: (b[0] + b[2]) / 2.0)

    teeth: list[dict] = []
    for idx, bbox in enumerate(bboxes_sorted):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # --- crest_line: bone line whose centroid is closest to bbox centre ---
        crest_line: list[list[float]] = []
        if bone_lines:
            best_bl_dist = math.inf
            best_bl = None
            for bl in bone_lines:
                if not bl:
                    continue
                mean_x = sum(p[0] for p in bl) / len(bl)
                mean_y = sum(p[1] for p in bl) / len(bl)
                d = _dist2(mean_x, mean_y, cx, cy)
                if d < best_bl_dist:
                    best_bl_dist = d
                    best_bl = bl
            if best_bl is not None:
                crest_line = [list(best_bl[0]), list(best_bl[-1])]

        # --- cej: CEJ points whose x is within [x1-60, x2+60] ---
        candidates = [p for p in cej_points if (x1 - 60) <= p[0] <= (x2 + 60)]
        if len(candidates) > 2:
            candidates = sorted(candidates, key=lambda p: abs(p[0] - cx))[:2]
        cej = sorted(candidates, key=lambda p: p[0])  # mesial → distal

        # --- apex: Apex_Points strictly inside the bbox ---
        apex_candidates = [
            p for p in apex_points if x1 <= p[0] <= x2 and y1 <= p[1] <= y2
        ]
        apex_best: list[list[float]] = []
        if apex_candidates:
            nearest = min(apex_candidates, key=lambda p: _dist2(p[0], p[1], cx, cy))
            apex_best = [list(nearest)]

        # Skip if both cej and crest_line are empty.
        if not cej and not crest_line:
            continue

        tooth: dict = {
            "tooth_id": str(idx + 1),
            "cej": [list(p) for p in cej],
            "crest_line": crest_line,
        }
        if apex_best:
            tooth["apex"] = apex_best

        teeth.append(tooth)

    return {"image": image_id, "teeth": teeth}


@dataclass(frozen=True)
class RealDenparRecord:
    """A single image record from the real DenPAR dataset."""

    image_id: str
    image_path: Path
    kp_path: Path  # Key Points Annotations JSON
    bl_path: Path  # Bone Level Annotations JSON
    annotation_dict: dict
    split: str  # "train" | "val" | "test"


class RealDenparAdapter:
    """Adapter for the real DenPAR dataset with the three-split layout.

    Parameters
    ----------
    root:
        Path to the ``Dataset/`` directory (contains ``Training/``,
        ``Testing/``, ``Validation/`` subdirectories).
    """

    _SPLIT_MAP: dict[str, str] = {
        "Training": "train",
        "Testing": "test",
        "Validation": "val",
    }

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def records(self, split: str | None = None) -> Iterator[RealDenparRecord]:
        """Yield :class:`RealDenparRecord`, optionally filtered by split.

        Parameters
        ----------
        split:
            One of ``"train"``, ``"val"``, ``"test"``.  If *None*, all
            splits are yielded.
        """
        for dir_name, split_label in self._SPLIT_MAP.items():
            if split is not None and split_label != split:
                continue
            yield from self._iter_split(dir_name, split_label)

    def _iter_split(
        self, split_name: str, split_label: str
    ) -> Iterator[RealDenparRecord]:
        """Yield records for one split directory.

        Parameters
        ----------
        split_name:
            Directory name inside ``self.root`` (e.g. ``"Training"``).
        split_label:
            Short label assigned to the split (e.g. ``"train"``).
        """
        split_dir = self.root / split_name
        images_dir = split_dir / "Images"
        kp_dir = split_dir / "Key Points Annotations"
        bl_dir = split_dir / "Bone Level Annotations"

        if not kp_dir.is_dir():
            return  # silently skip missing split directories

        for kp_path in sorted(kp_dir.glob("*.json")):
            stem = kp_path.stem  # e.g. "1001"
            bl_path = bl_dir / kp_path.name
            if not bl_path.exists():
                continue  # no bone-level annotation → skip

            kp_data = json.loads(kp_path.read_text(encoding="utf-8"))
            bl_data = json.loads(bl_path.read_text(encoding="utf-8"))

            annotation_dict = _build_annotation_dict(kp_data, bl_data)

            image_name = kp_data.get("Image_id", f"{stem}.jpg")
            image_path = images_dir / image_name

            yield RealDenparRecord(
                image_id=stem,
                image_path=image_path,
                kp_path=kp_path,
                bl_path=bl_path,
                annotation_dict=annotation_dict,
                split=split_label,
            )


def load_real_denpar(
    root: Path | str, split: str | None = None
) -> list[RealDenparRecord]:
    """Return a list of all :class:`RealDenparRecord` from the given DenPAR root.

    Parameters
    ----------
    root:
        Path to the ``Dataset/`` directory.
    split:
        Optional split filter: ``"train"``, ``"val"``, or ``"test"``.

    Returns
    -------
    list[RealDenparRecord]
    """
    return list(RealDenparAdapter(root).records(split=split))
