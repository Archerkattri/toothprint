"""Longitudinal pairing utilities for phone-capture DatasetRecords.

Pairs records from the same subject across different timepoints so that
downstream pipelines can compare t0 vs t1 captures without manual bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LongitudinalPair:
    """A matched pair of DatasetRecords from the same subject at two timepoints.

    Attributes:
        subject_id: Shared subject identifier extracted from the record notes.
        t0_record:  The earlier (baseline) ``DatasetRecord``.
        t1_record:  The later (follow-up) ``DatasetRecord``.
    """

    subject_id: str
    t0_record: Any  # DatasetRecord
    t1_record: Any  # DatasetRecord


def _subject_id_from_record(record: Any) -> str | None:
    """Extract subject_id from a DatasetRecord.

    PhoneCaptureLoader embeds ``subject=<id>`` in ``record.notes``.
    If that field is absent we fall back to splitting ``record_id`` on ``_``
    and using the second token (e.g. ``phonecap_subj001_t0_anterior`` → ``subj001``).
    """
    notes: str = getattr(record, "notes", "") or ""
    for part in notes.split():
        if part.startswith("subject="):
            return part[len("subject="):]

    record_id: str = getattr(record, "record_id", "") or ""
    tokens = record_id.split("_")
    # phonecap_<subject>_<timepoint>_<stem>  has at least 3 tokens
    if len(tokens) >= 3 and tokens[0] == "phonecap":
        return tokens[1]

    return None


def _timepoint_sort_key(timepoint: str) -> tuple[int, object]:
    """Natural-ordering key for timepoint labels.

    Sorts by the trailing integer when present (so ``t2`` < ``t10``), falling
    back to lexicographic order for non-numeric labels. Numeric labels always
    sort before purely-string ones for a stable total order.
    """
    digits = "".join(ch for ch in timepoint if ch.isdigit())
    if digits:
        return (0, int(digits))
    return (1, timepoint)


def _timepoint_from_record(record: Any) -> str:
    """Extract timepoint label from a DatasetRecord.

    PhoneCaptureLoader embeds ``timepoint=<id>`` in ``record.notes``.
    Falls back to the third ``_``-delimited token of ``record_id``.
    """
    notes: str = getattr(record, "notes", "") or ""
    for part in notes.split():
        if part.startswith("timepoint="):
            return part[len("timepoint="):]

    record_id: str = getattr(record, "record_id", "") or ""
    tokens = record_id.split("_")
    if len(tokens) >= 3:
        return tokens[2]

    return ""


def pair_by_subject(records: list[Any]) -> list[LongitudinalPair]:
    """Match records with the same subject_id across different timepoints.

    Groups records by ``subject_id`` (extracted from ``notes`` or
    ``record_id``), then for each subject with at least two distinct
    timepoints pairs the lexicographically-earliest timepoint (t0) against
    the lexicographically-latest timepoint (t1).

    Records whose subject_id cannot be determined are silently skipped.
    Subjects with only a single timepoint are also skipped (nothing to pair).

    Args:
        records: List of ``DatasetRecord`` instances (typically from
            ``PhoneCaptureLoader.records()``).

    Returns:
        List of ``LongitudinalPair`` objects, one per subject that has at
        least two distinct timepoints.  The order follows the sorted order
        of ``subject_id``.
    """
    # Group records by subject_id then by timepoint_id.
    from collections import defaultdict

    # subject_id -> timepoint_id -> list[record]
    grouped: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))

    for record in records:
        subject_id = _subject_id_from_record(record)
        if subject_id is None:
            continue
        timepoint = _timepoint_from_record(record)
        grouped[subject_id][timepoint].append(record)

    pairs: list[LongitudinalPair] = []
    for subject_id in sorted(grouped.keys()):
        # Natural-sort so "t2" precedes "t10" (lexicographic order mis-pairs
        # the earliest/latest timepoints once counts reach double digits).
        timepoints = sorted(grouped[subject_id].keys(), key=_timepoint_sort_key)
        if len(timepoints) < 2:
            continue  # Need at least two timepoints to form a pair.
        t0_tp = timepoints[0]
        t1_tp = timepoints[-1]
        # Use the first record at each timepoint as the representative.
        t0_record = grouped[subject_id][t0_tp][0]
        t1_record = grouped[subject_id][t1_tp][0]
        pairs.append(LongitudinalPair(subject_id=subject_id, t0_record=t0_record, t1_record=t1_record))

    return pairs
