"""Validate split integrity for the perio-KPT dataset.

Checks for image_id collisions between the experiment/baseline and holdout
splits, computes checksums for all records, and saves them to a JSON file.

Usage::

    python scripts/validate_splits.py
    python scripts/validate_splits.py --root data/perio-kpt/extracted/perio_KPT
    python scripts/validate_splits.py --checksums outputs/split_checksums.json
    python scripts/validate_splits.py --verify outputs/split_checksums.json

Exit codes:
    0 — no leakage detected and checksums written/verified successfully.
    1 — leakage found or checksum verification failed or data root missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcc.data.perio_kpt_adapter import PerioKptAdapter
from dcc.splits.checksums import (
    check_split_leakage,
    freeze_split_checksums,
    verify_split_checksums,
)

_DEFAULT_ROOT = Path("data/perio-kpt/extracted/perio_KPT")
_DEFAULT_CHECKSUMS = Path("outputs/split_checksums.json")


def run(
    root: Path,
    checksums_path: Path,
    verify: bool,
) -> int:
    if not root.exists():
        print(
            f"ERROR: perio-KPT root not found at {root}. "
            "Download and extract the dataset first.",
            file=sys.stderr,
        )
        return 1

    adapter = PerioKptAdapter(root)

    # Load all records grouped by split for leakage detection.
    # We treat "experiment" + "baseline" as "train" and "holdout" as "test".
    train_records: list = []
    test_records: list = []
    all_records: list = []

    for record in adapter.records():
        all_records.append(record)
        if record.split in {"baseline", "experiment"}:
            train_records.append(record)
        elif record.split == "holdout":
            test_records.append(record)

    print(f"Loaded {len(all_records)} total records.")
    print(f"  Train (baseline + experiment): {len(train_records)}")
    print(f"  Test  (holdout):               {len(test_records)}")
    print()

    # --- Leakage check ---
    leaked = check_split_leakage(train_records, test_records)
    if leaked:
        print(f"LEAKAGE DETECTED: {len(leaked)} image_id(s) appear in both splits:")
        for image_id in leaked:
            print(f"  {image_id}")
        leakage_ok = False
    else:
        print("No leakage detected between train and test splits.")
        leakage_ok = True
    print()

    # --- Checksum verification or freeze ---
    if verify:
        if not checksums_path.exists():
            print(
                f"ERROR: Checksum file not found at {checksums_path}. "
                "Run without --verify to create it first.",
                file=sys.stderr,
            )
            return 1

        print(f"Verifying checksums against {checksums_path} ...")
        ok = verify_split_checksums(all_records, checksums_path)
        if ok:
            print("All checksums match.")
        else:
            print("CHECKSUM MISMATCH: some records changed since the file was frozen.")
        print()
        return 0 if (ok and leakage_ok) else 1
    else:
        written = freeze_split_checksums(all_records, checksums_path)
        print(f"Checksums for {len(all_records)} records written to {written}.")
        print()
        return 0 if leakage_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate perio-KPT split integrity and save checksums."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_DEFAULT_ROOT,
        help="Path to the extracted perio_KPT directory",
    )
    parser.add_argument(
        "--checksums",
        type=Path,
        default=_DEFAULT_CHECKSUMS,
        metavar="PATH",
        help="Path for the checksum JSON file (written or read)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Verify existing checksums instead of writing new ones",
    )
    args = parser.parse_args(argv)
    return run(root=args.root, checksums_path=args.checksums, verify=args.verify)


if __name__ == "__main__":
    raise SystemExit(main())
