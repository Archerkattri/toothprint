"""Deterministic split freezing.

The benchmark cannot tune thresholds after seeing the test set. This module makes
that rule explicit by turning a list of case identifiers into a persisted split
manifest with stable hashing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SplitAssignment:
    case_id: str
    split: str
    stratum: str = "default"


@dataclass(frozen=True)
class FrozenSplit:
    schema_version: str
    seed: str
    assignments: list[SplitAssignment]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for item in self.assignments:
            out[item.split] = out.get(item.split, 0) + 1
        return out

    def ids_for(self, split: str) -> list[str]:
        return [item.case_id for item in self.assignments if item.split == split]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "counts": self.counts(),
            "assignments": [asdict(item) for item in self.assignments],
        }


def build_deterministic_split(
    case_ids: Iterable[str],
    *,
    seed: str = "dental-change-cert-v0",
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
    stratum: str = "default",
) -> FrozenSplit:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError("train + calibration fractions must leave a test split")

    unique_ids = sorted(set(case_ids))
    if len(unique_ids) < 3:
        raise ValueError("at least three cases are required to freeze train/calibration/test splits")

    ranked = sorted(unique_ids, key=lambda case_id: _score(seed, case_id))
    n = len(ranked)
    train_end = max(1, int(round(n * train_fraction)))
    cal_end = max(train_end + 1, int(round(n * (train_fraction + calibration_fraction))))
    cal_end = min(cal_end, n - 1)

    assignments: list[SplitAssignment] = []
    for i, case_id in enumerate(ranked):
        if i < train_end:
            split = "train"
        elif i < cal_end:
            split = "calibration"
        else:
            split = "test"
        assignments.append(SplitAssignment(case_id=case_id, split=split, stratum=stratum))

    return FrozenSplit(schema_version="0.1", seed=seed, assignments=sorted(assignments, key=lambda x: x.case_id))


def write_split(split: FrozenSplit, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _score(seed: str, case_id: str) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()

