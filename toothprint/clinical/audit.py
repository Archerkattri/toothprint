"""Append-only audit trail — every certificate is traceable.

Medical software must record what was decided, on which input, under which
calibration, and when. Each certificate appends an immutable record; the log can
be serialised to JSONL for the patient/forensic record.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def input_fingerprint(array) -> str:
    """SHA-256 of an input array — links a decision to the exact capture."""
    arr = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(arr.tobytes()).hexdigest()


@dataclass(frozen=True)
class AuditRecord:
    timestamp_utc: str
    input_fingerprint: str
    calibration_id: str
    decision: str
    measured: float
    interval: tuple
    operator: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["interval"] = list(self.interval)
        return d


class AuditLog:
    """In-memory append-only log with JSONL export."""

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    def record(self, rec: AuditRecord) -> None:
        self._records.append(rec)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def to_jsonl(self, path: "str | Path") -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for r in self._records:
                fh.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")
        return p
