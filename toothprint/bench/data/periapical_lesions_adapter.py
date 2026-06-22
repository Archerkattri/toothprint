"""Loader for the periapical lesions classification dataset (Zenodo 13772918).

Dataset structure::

    root/
      1. Rx Canon/    *.JPG   (images 1.N.JPG)
      2. Rx iPhone/   *.JPG   (images 2.N.JPG)
      3. Rx Xiaomi/   *.jpg   (images 3.N.jpg)
      periapical_lesions_classification.xlsx

The Excel file maps Rx N° (image number) to:
  - tooth (Spanish name)
  - lesion_present: True when label column == 'L', False when 'SL'

This is an endodontic classification dataset (periapical lesion presence / absence),
not a bone-level annotation dataset. It does not feed into the conformal bone-level
scoring pipeline directly, but can be used for:
  - Auxiliary lesion classifier training
  - Image quality / device diversity benchmarks
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_CAMERA_DIRS = {
    "canon": ("1. Rx Canon", "1."),
    "iphone": ("2. Rx iPhone", "2."),
    "xiaomi": ("3. Rx Xiaomi", "3."),
}


@dataclass(frozen=True)
class PeriapicalRecord:
    image_id: str
    image_path: Path
    rx_number: int
    tooth_name: str
    lesion_present: bool
    camera: str
    split: str


def _hash_split(key: str, train_frac: float = 0.7, val_frac: float = 0.15) -> str:
    digest = int(hashlib.md5(key.encode()).hexdigest(), 16)
    bucket = (digest % 100) / 100.0
    if bucket < train_frac:
        return "train"
    if bucket < train_frac + val_frac:
        return "val"
    return "test"


class PeriapicalLesionsAdapter:
    """Parse the periapical lesions dataset into PeriapicalRecord instances.

    Parameters
    ----------
    root:
        Path to the extracted dataset root containing the Excel file and
        camera-specific subdirectories.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def records(
        self,
        split: str | None = None,
        camera: str | None = None,
    ) -> Iterator[PeriapicalRecord]:
        """Yield records, optionally filtered by split and/or camera.

        Parameters
        ----------
        split:
            One of "train", "val", "test" or None for all.
        camera:
            One of "canon", "iphone", "xiaomi" or None for all.
        """
        labels = self._load_labels()
        for rec in self._iter_all(labels):
            if split is not None and rec.split != split:
                continue
            if camera is not None and rec.camera != camera:
                continue
            yield rec

    def _load_labels(self) -> dict[int, tuple[str, bool]]:
        """Return {rx_number: (tooth_name, lesion_present)} from the Excel file."""
        xlsx_path = self.root / "periapical_lesions_classification.xlsx"
        if not xlsx_path.exists():
            return {}
        try:
            import openpyxl
        except ImportError:
            return {}

        wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
        ws = wb.active
        labels: dict[int, tuple[str, bool]] = {}
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                continue  # header
            rx_num, tooth, label_raw = row[0], row[1], row[2]
            if not isinstance(rx_num, int):
                continue
            tooth_name = str(tooth) if tooth else ""
            lesion = str(label_raw).strip().upper() == "L" if label_raw else False
            labels[rx_num] = (tooth_name, lesion)
        wb.close()
        return labels

    def _iter_all(self, labels: dict) -> Iterator[PeriapicalRecord]:
        for cam_key, (dir_name, prefix) in _CAMERA_DIRS.items():
            cam_dir = self.root / dir_name
            if not cam_dir.is_dir():
                continue
            for img_path in sorted(cam_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                # Parse Rx number from filename: "1.37.JPG" → prefix="1." → rx=37
                stem = img_path.stem
                if not stem.startswith(prefix):
                    continue
                try:
                    rx_num = int(stem[len(prefix) :])
                except ValueError:
                    continue
                tooth_name, lesion = labels.get(rx_num, ("", False))
                image_id = f"{cam_key}_{rx_num:04d}"
                split = _hash_split(image_id)
                yield PeriapicalRecord(
                    image_id=image_id,
                    image_path=img_path,
                    rx_number=rx_num,
                    tooth_name=tooth_name,
                    lesion_present=lesion,
                    camera=cam_key,
                    split=split,
                )


def load_periapical_lesions(
    root: Path | str,
    split: str | None = None,
    camera: str | None = None,
) -> list[PeriapicalRecord]:
    """Return all PeriapicalRecord instances from the given dataset root."""
    return list(PeriapicalLesionsAdapter(root).records(split=split, camera=camera))
