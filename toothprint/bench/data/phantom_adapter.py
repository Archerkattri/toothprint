"""Phantom / typodont repeat-acquisition dataset adapter.

A phantom or typodont is a plastic dental model photographed multiple times
under different acquisition conditions (angles, exposure, repositioning).
Ground truth: ALL pairs are stable (no change). This makes phantoms ideal
for measuring the false-positive rate in a controlled setting without
confounders from real pathology.

Expected directory layout:
    phantom_root/
        phantom_001/
            acq_000.json    ← annotation_dict (DenPAR-style)
            acq_001.json
            acq_002.json
            ...
        phantom_002/
            ...
        metadata.json       ← optional: {"phantom_id": ..., "material": ...}

Each .json file should contain a DenPAR-style annotation dict with "teeth" list.
If the directory is absent or empty, the adapter yields no records.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhantomRecord:
    image_id: str
    phantom_id: str
    acquisition_idx: int
    annotation_dict: dict
    material: str = "unknown"  # "plastic", "resin", "wax", etc.


class PhantomAdapter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def records(self) -> list[PhantomRecord]:
        """Yield all phantom acquisition records."""
        if not self.root.exists():
            return []
        records = []
        for phantom_dir in sorted(self.root.iterdir()):
            if not phantom_dir.is_dir():
                continue
            phantom_id = phantom_dir.name
            # Load optional metadata
            meta_path = phantom_dir / "metadata.json"
            material = "unknown"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    material = meta.get("material", "unknown")
                except Exception:
                    pass
            # Load acquisition .json files
            acq_files = sorted(phantom_dir.glob("acq_*.json"))
            for idx, acq_path in enumerate(acq_files):
                try:
                    ann = json.loads(acq_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                image_id = f"{phantom_id}_acq_{idx:03d}"
                records.append(
                    PhantomRecord(
                        image_id=image_id,
                        phantom_id=phantom_id,
                        acquisition_idx=idx,
                        annotation_dict=ann,
                        material=material,
                    )
                )
        return records

    def stable_pairs(self) -> list:
        """Return all between-acquisition pairs within each phantom.

        All pairs are labelled "stable" (no real change — it's a plastic model).
        Returns list of dicts {"baseline": ann_dict, "followup": ann_dict,
                               "label": "stable", "phantom_id": str, "true_change": 0.0}
        """
        from itertools import combinations

        by_phantom: dict[str, list[PhantomRecord]] = {}
        for rec in self.records():
            by_phantom.setdefault(rec.phantom_id, []).append(rec)

        pairs = []
        for phantom_id, recs in by_phantom.items():
            for r1, r2 in combinations(recs, 2):
                pairs.append(
                    {
                        "baseline": r1.annotation_dict,
                        "followup": r2.annotation_dict,
                        "label": "stable",
                        "phantom_id": phantom_id,
                        "true_change": 0.0,
                    }
                )
        return pairs


def load_phantom_data(root: str | Path) -> list[PhantomRecord]:
    """Convenience function: load all phantom records from root."""
    return PhantomAdapter(root).records()
