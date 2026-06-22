"""perio-KPT YOLO-keypoint → DenPAR-style annotation adapter.

The perio-KPT dataset uses YOLO keypoint format (38 values per line):
    class cx cy bw bh kp1x kp1y v1 kp2x kp2y v2 ... kp11x kp11y v11

Keypoint semantics:
    kp1, kp2  — CEJ left/right endpoints
    kp4, kp5  — alveolar bone crest mesial/distal endpoints
    kp7       — root apex tip

Visibility convention: v=0 → unlabeled (skip), v>=1 → valid point.

This adapter converts to DenPAR-compatible annotation dicts with pixel coords
so the existing scoring pipeline (`dcc.score.periodontal`) can be used directly.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Keypoint index → 0-based index into the 11 triplets (x, y, v)
_KP_CEJ_LEFT = 0  # kp1
_KP_CEJ_RIGHT = 1  # kp2
_KP_CREST_MESIAL = 3  # kp4
_KP_CREST_DISTAL = 4  # kp5
_KP_APEX = 6  # kp7


@dataclass(frozen=True)
class PerioKptRecord:
    image_id: str
    image_path: Path
    label_path: Path
    annotation_dict: dict
    split: str  # "baseline" | "experiment" | "holdout" | "external"


class PerioKptAdapter:
    """Parse perio-KPT YOLO keypoint labels into DenPAR-compatible dicts.

    Expected root layout::

        root/
          0_Baseline/
            images/   *.png
            labels/   *.txt
          1_Experiment/
            standard_box/
              f0/train/{images,labels}/
              f0/val/{images,labels}/
              ...
            holdout_test_standard_box/
              images/
              labels/
          3_External_Set/
            standard_box/
              images/
              labels/
    """

    def __init__(self, root_dir: Path | str) -> None:
        self.root = Path(root_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def records(self, split: str | None = None) -> Iterator[PerioKptRecord]:
        """Yield PerioKptRecord instances, optionally filtered by split name."""
        for record in self._iter_all():
            if split is None or record.split == split:
                yield record

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _iter_all(self) -> Iterator[PerioKptRecord]:
        yield from self._iter_baseline()
        yield from self._iter_experiment()
        yield from self._iter_holdout()
        yield from self._iter_external()

    def _iter_baseline(self) -> Iterator[PerioKptRecord]:
        images_dir = self.root / "0_Baseline" / "images"
        labels_dir = self.root / "0_Baseline" / "labels"
        if not images_dir.is_dir():
            return
        yield from self._iter_image_label_dir(images_dir, labels_dir, split="baseline")

    def _iter_experiment(self) -> Iterator[PerioKptRecord]:
        """Iterate experiment images from all fold train/val sub-dirs.

        De-duplicates images seen multiple times across folds by tracking the
        image stem so each physical image is only yielded once.
        """
        base = self.root / "1_Experiment" / "standard_box"
        if not base.is_dir():
            return
        seen: set[str] = set()
        for fold_dir in sorted(base.iterdir()):
            if not fold_dir.is_dir() or not fold_dir.name.startswith("f"):
                continue
            for subset in ("train", "val"):
                images_dir = fold_dir / subset / "images"
                labels_dir = fold_dir / subset / "labels"
                if not images_dir.is_dir():
                    continue
                for record in self._iter_image_label_dir(
                    images_dir, labels_dir, split="experiment"
                ):
                    if record.image_id not in seen:
                        seen.add(record.image_id)
                        yield record

    def _iter_holdout(self) -> Iterator[PerioKptRecord]:
        images_dir = self.root / "1_Experiment" / "holdout_test_standard_box" / "images"
        labels_dir = self.root / "1_Experiment" / "holdout_test_standard_box" / "labels"
        if not images_dir.is_dir():
            return
        yield from self._iter_image_label_dir(images_dir, labels_dir, split="holdout")

    def _iter_external(self) -> Iterator[PerioKptRecord]:
        images_dir = self.root / "3_External_Set" / "standard_box" / "images"
        labels_dir = self.root / "3_External_Set" / "standard_box" / "labels"
        if not images_dir.is_dir():
            return
        yield from self._iter_image_label_dir(images_dir, labels_dir, split="external")

    def _iter_image_label_dir(
        self,
        images_dir: Path,
        labels_dir: Path,
        split: str,
    ) -> Iterator[PerioKptRecord]:
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            label_path = labels_dir / (image_path.stem + ".txt")
            if not label_path.exists():
                continue

            width, height = _read_image_size(image_path)
            if width == 0 or height == 0:
                continue

            teeth = _parse_yolo_label(label_path, width, height)
            if not teeth:
                continue

            annotation = {
                "image": image_path.name,
                "teeth": teeth,
            }
            yield PerioKptRecord(
                image_id=image_path.stem,
                image_path=image_path,
                label_path=label_path,
                annotation_dict=annotation,
                split=split,
            )


# ------------------------------------------------------------------
# YOLO label parsing
# ------------------------------------------------------------------


def _parse_yolo_label(label_path: Path, width: int, height: int) -> list[dict]:
    """Return DenPAR-style teeth list from a YOLO keypoint label file."""
    text = label_path.read_text(encoding="utf-8")
    teeth: list[dict] = []
    tooth_counter = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        tooth = _parse_yolo_line(line, width, height, tooth_counter)
        if tooth is None:
            continue
        teeth.append(tooth)
        tooth_counter += 1
    return teeth


def _parse_yolo_line(line: str, width: int, height: int, index: int) -> dict | None:
    """Parse a single YOLO keypoint line and return a DenPAR tooth dict or None."""
    parts = line.split()
    # Expect: class cx cy bw bh + 11 keypoints × 3 values = 5 + 33 = 38 fields
    if len(parts) < 38:
        return None

    # Parts: [0]=class [1..4]=bbox [5..37]=11 keypoints (x y v each)
    kps: list[tuple[float, float, float]] = []
    for k in range(11):
        offset = 5 + k * 3
        kx = float(parts[offset])
        ky = float(parts[offset + 1])
        vis = float(parts[offset + 2])
        kps.append((kx, ky, vis))

    def to_px(nx: float, ny: float) -> list[float]:
        return [round(nx * width, 6), round(ny * height, 6)]

    def valid_point(idx: int) -> list[float] | None:
        kx, ky, vis = kps[idx]
        if vis <= 0.0 or (kx == 0.0 and ky == 0.0):
            return None
        return to_px(kx, ky)

    cej_left = valid_point(_KP_CEJ_LEFT)
    cej_right = valid_point(_KP_CEJ_RIGHT)
    crest_mesial = valid_point(_KP_CREST_MESIAL)
    crest_distal = valid_point(_KP_CREST_DISTAL)
    apex = valid_point(_KP_APEX)

    cej = [p for p in [cej_left, cej_right] if p is not None]
    crest_line = [p for p in [crest_mesial, crest_distal] if p is not None]

    # Skip teeth where both cej and crest_line are empty
    if not cej and not crest_line:
        return None

    tooth: dict = {
        "tooth_id": str(index + 1),
        "cej": cej,
        "crest_line": crest_line,
    }
    if apex is not None:
        tooth["apex"] = [apex]

    return tooth


# ------------------------------------------------------------------
# Image size reading (stdlib only; PIL not required)
# ------------------------------------------------------------------


def _read_image_size(image_path: Path) -> tuple[int, int]:
    """Return (width, height) for PNG or JPEG without external dependencies."""
    suffix = image_path.suffix.lower()
    try:
        if suffix == ".png":
            return _png_size(image_path)
        if suffix in {".jpg", ".jpeg"}:
            return _jpeg_size(image_path)
    except Exception:
        pass
    return 0, 0


def _png_size(path: Path) -> tuple[int, int]:
    """Read PNG dimensions from the IHDR chunk header (bytes 16-24)."""
    with open(path, "rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            return 0, 0
        f.read(4)  # IHDR chunk length
        f.read(4)  # "IHDR"
        w_bytes = f.read(4)
        h_bytes = f.read(4)
    w = struct.unpack(">I", w_bytes)[0]
    h = struct.unpack(">I", h_bytes)[0]
    return w, h


def _jpeg_size(path: Path) -> tuple[int, int]:
    """Read JPEG dimensions by scanning for SOF markers."""
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    if data[0:2] != b"\xff\xd8":
        return 0, 0
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        i += 2
        # SOF markers: 0xC0-0xC3, 0xC5-0xC7, 0xC9-0xCB, 0xCD-0xCF
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            # length (2) + precision (1) + height (2) + width (2)
            if i + 6 > len(data):
                break
            h = struct.unpack(">H", data[i + 3 : i + 5])[0]
            w = struct.unpack(">H", data[i + 5 : i + 7])[0]
            return w, h
        # Skip this segment
        if i + 1 >= len(data):
            break
        seg_len = struct.unpack(">H", data[i : i + 2])[0]
        i += seg_len
    return 0, 0
