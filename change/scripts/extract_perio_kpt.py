"""Extract perio-KPT records into a manifest JSON.

Usage::

    python scripts/extract_perio_kpt.py \\
        --limit 10 \\
        --output outputs/perio_kpt_manifest.json

The manifest is a JSON array of objects with the following fields per image:
    image_id        — stem of the image filename
    image_path      — absolute path to the image
    annotation_path — None (YOLO labels inline; no separate JSON)
    split           — "baseline" | "experiment" | "holdout" | "external"
    tooth_count     — total teeth parsed from the label
    has_cej         — True if at least one tooth has a non-empty cej list
    has_crest       — True if at least one tooth has a non-empty crest_line list
    has_apex        — True if at least one tooth has an apex entry
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcc.data.perio_kpt_adapter import PerioKptAdapter
from dcc.score.periodontal import record_change_scores


_DEFAULT_ROOT = Path("data/perio-kpt/extracted/perio_KPT")
_DEFAULT_OUTPUT = Path("outputs/perio_kpt_manifest.json")


def run(root: Path, output: Path, limit: int | None) -> int:
    if not root.exists():
        print(
            f"WARNING: perio-KPT root not found at {root}. "
            "Download and extract the dataset first.",
            file=sys.stderr,
        )
        return 1

    adapter = PerioKptAdapter(root)

    entries: list[dict] = []
    n_images = 0
    n_teeth = 0
    n_complete = 0  # teeth with CEJ + crest + apex

    for record in adapter.records():
        if limit is not None and n_images >= limit:
            break

        # Validate that record_change_scores runs without error.
        # We use the annotation against itself (diff = 0) to exercise the pipeline.
        ann = record.annotation_dict
        try:
            record_change_scores(ann, ann)
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING: record_change_scores failed for {record.image_id}: {exc}",
                file=sys.stderr,
            )

        teeth = ann.get("teeth", [])
        has_cej = any(bool(t.get("cej")) for t in teeth)
        has_crest = any(bool(t.get("crest_line")) for t in teeth)
        has_apex = any(bool(t.get("apex")) for t in teeth)

        tooth_complete = sum(
            1
            for t in teeth
            if bool(t.get("cej")) and bool(t.get("crest_line")) and bool(t.get("apex"))
        )

        entries.append(
            {
                "image_id": record.image_id,
                "image_path": str(record.image_path),
                "annotation_path": None,
                "split": record.split,
                "tooth_count": len(teeth),
                "has_cej": has_cej,
                "has_crest": has_crest,
                "has_apex": has_apex,
            }
        )

        n_images += 1
        n_teeth += len(teeth)
        n_complete += tooth_complete

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")

    print(f"Processed: {n_images} images")
    print(f"Total teeth: {n_teeth}")
    print(f"Teeth with complete CEJ+crest+apex: {n_complete}")
    print(f"Manifest written to: {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract perio-KPT records into a manifest JSON."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_DEFAULT_ROOT,
        help="Path to the extracted perio_KPT directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output manifest JSON path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only process first N images (for fast validation)",
    )
    args = parser.parse_args(argv)
    return run(root=args.root, output=args.output, limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
