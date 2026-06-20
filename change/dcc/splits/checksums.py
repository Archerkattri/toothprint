"""Split integrity checks for DentalChangeCert.

Functions here let you:
* Compute a deterministic SHA-256 checksum per PerioKptRecord.
* Detect image_id collisions between train and test splits.
* Freeze checksums to a JSON file and later verify them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def compute_record_checksum(image_path: Path, label_path: Path) -> str:
    """SHA-256 of concatenated image + label bytes.

    Parameters
    ----------
    image_path:
        Path to the image file whose bytes are hashed.
    label_path:
        Path to the label/annotation file whose bytes are hashed.

    Returns
    -------
    str
        Lowercase hex SHA-256 digest of ``image_path`` bytes + ``label_path`` bytes.
    """
    h = hashlib.sha256()
    h.update(Path(image_path).read_bytes())
    h.update(Path(label_path).read_bytes())
    return h.hexdigest()


def check_split_leakage(
    train_records: list,
    test_records: list,
) -> list[str]:
    """Return image_ids that appear in both the train and test record lists.

    Parameters
    ----------
    train_records:
        Records assigned to the training split.
    test_records:
        Records assigned to the test/holdout split.

    Returns
    -------
    list[str]
        Sorted list of image_ids present in both splits.  An empty list
        means no leakage was detected.
    """
    train_ids = {r.image_id for r in train_records}
    test_ids = {r.image_id for r in test_records}
    leaked = train_ids & test_ids
    return sorted(leaked)


def freeze_split_checksums(records: list, output_path: Path) -> Path:
    """Write a JSON file with a checksum for each record.

    Each record must expose ``.image_id``, ``.image_path``, and ``.label_path``.
    If a record does not have ``.label_path``, ``.image_path`` is used for
    both fields (so only image bytes contribute to the digest).

    The output format is::

        {
          "schema_version": "0.1",
          "records": [
            {"image_id": "Image100", "checksum": "<hex>"},
            ...
          ]
        }

    Records are sorted by ``image_id`` for stable diffs.

    Parameters
    ----------
    records:
        Iterable of objects exposing ``.image_id``, ``.image_path``, and
        optionally ``.label_path``.
    output_path:
        Destination path for the JSON file.  Parent directories are created
        if they don't exist.

    Returns
    -------
    Path
        The resolved ``output_path`` that was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import warnings as _warnings

    def _label(r) -> Path:
        lp = getattr(r, "label_path", None)
        if lp is None:
            _warnings.warn(
                f"Record {r.image_id!r} has no label_path; checksum covers image bytes only. "
                "Tampered label files will not be detected.",
                UserWarning,
                stacklevel=4,
            )
            return Path(r.image_path)
        return Path(lp)

    entries = [
        {
            "image_id": r.image_id,
            "checksum": compute_record_checksum(Path(r.image_path), _label(r)),
        }
        for r in records
    ]
    entries.sort(key=lambda e: e["image_id"])

    payload = {
        "schema_version": "0.1",
        "records": entries,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def verify_split_checksums(records: list, checksum_path: Path) -> bool:
    """Verify that every record matches its saved checksum.

    Parameters
    ----------
    records:
        The records to verify (same order or any order—matched by image_id).
        Each record must expose ``.image_id``, ``.image_path``, and optionally
        ``.label_path``.
    checksum_path:
        Path to a JSON file previously written by :func:`freeze_split_checksums`.

    Returns
    -------
    bool
        ``True`` if every record matches its saved checksum AND every saved
        checksum has a corresponding record.  ``False`` otherwise.
    """
    checksum_path = Path(checksum_path)
    payload = json.loads(checksum_path.read_text(encoding="utf-8"))
    saved: dict[str, str] = {
        entry["image_id"]: entry["checksum"]
        for entry in payload.get("records", [])
    }

    current: dict[str, str] = {
        r.image_id: compute_record_checksum(
            Path(r.image_path),
            Path(getattr(r, "label_path", r.image_path)),
        )
        for r in records
    }

    if set(saved.keys()) != set(current.keys()):
        return False

    return all(saved[image_id] == current[image_id] for image_id in saved)
