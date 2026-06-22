"""Frozen train/val/test split registry.

Splits are committed at dataset ingest time and never re-shuffled after
any calibration or threshold tuning. This prevents the oracle leak where
a researcher tunes tau on data that later appears in eval.

The registry uses a deterministic hash of (image_id, salt) to assign splits.
"""

from __future__ import annotations
import hashlib
from enum import Enum


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def assign_split(
    image_id: str,
    *,
    salt: str = "dcc_v1",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Split:
    """Deterministically assign a split label to an image_id.

    Uses SHA-256 of (image_id + salt) → first 8 hex digits → int → [0,1].
    Splits are frozen: same image_id + salt always returns the same split.
    """
    digest = hashlib.sha256(f"{image_id}:{salt}".encode()).hexdigest()
    u = int(digest[:8], 16) / 0xFFFFFFFF
    if u < train_frac:
        return Split.TRAIN
    elif u < train_frac + val_frac:
        return Split.VAL
    else:
        return Split.TEST


def split_records(
    records: list,
    *,
    salt: str = "dcc_v1",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    id_attr: str = "image_id",
) -> dict[str, list]:
    """Assign all records to splits and return {split_name: [records]}.

    Reads image_id from getattr(record, id_attr) or record[id_attr].
    """
    result: dict[str, list] = {s.value: [] for s in Split}
    for rec in records:
        try:
            image_id = getattr(rec, id_attr)
        except AttributeError:
            image_id = rec[id_attr]
        split = assign_split(
            image_id, salt=salt, train_frac=train_frac, val_frac=val_frac
        )
        result[split.value].append(rec)
    return result
