#!/usr/bin/env python3
"""DentalMapCert coverage evaluation using 3DTeethLand landmark point clouds.

3DTeethLand provides 3D landmark coordinates (Mesial, Distal, Cusp, InnerPoint,
OuterPoint, FacialPoint) for upper and lower dental arches from IOS scans.

This script loads landmark JSON files, extracts all 3D points per case+arch,
and evaluates coverage_from_point_cloud as a sanity check that the pipeline
works with real 3D dental geometry (sparse landmark clouds).

3DTeethLand: landmark annotation dataset (OSF um96h), training split.

Usage:
    python scripts/run_dmc_teethland.py \\
        --data data/teeth3ds/extracted \\
        --output outputs/dmc_teethland \\
        [--limit 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from dentalmapcert.coverage import coverage_from_point_cloud
from dentalmapcert.dataset_loaders import TeethLandLoader, load_teethland_points


def main() -> None:
    parser = argparse.ArgumentParser(description="DMC coverage eval on 3DTeethLand landmarks")
    parser.add_argument("--data", default="data/teeth3ds/extracted",
                        help="Root of 3DTeethLand extracted directory (contains upper/ and lower/)")
    parser.add_argument("--output", default="outputs/dmc_teethland",
                        help="Output directory for results JSON")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of records to process (None = all)")
    parser.add_argument("--grid", type=int, default=5,
                        help="Fixed voxel-grid resolution for comparable coverage (default: 5)")
    args = parser.parse_args()

    root = Path(args.data)
    if not root.exists():
        print(f"ERROR: 3DTeethLand data not found at {root}", file=sys.stderr)
        sys.exit(1)

    loader = TeethLandLoader(str(root))
    records = list(loader.records())
    print(f"Found {len(records)} 3DTeethLand records (upper + lower arches)")

    if args.limit:
        records = records[: args.limit]
        print(f"Processing first {args.limit} records")

    results = []
    coverage_fractions = []
    landmark_counts_by_class: dict[str, int] = {}

    for i, rec in enumerate(records):
        pts = load_teethland_points(rec)
        if pts.shape[0] == 0:
            print(f"  [{i+1}/{len(records)}] {rec.record_id}: skipped (empty landmark file)")
            continue

        # Parse landmark classes from JSON for statistics
        if rec.label_path:
            try:
                data = json.loads(Path(rec.label_path).read_text(encoding="utf-8"))
                for obj in data.get("objects", []):
                    cls = obj.get("class", "unknown")
                    landmark_counts_by_class[cls] = landmark_counts_by_class.get(cls, 0) + 1
            except Exception:
                pass

        # Build bounding box from landmarks with 5mm margin
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        margin = 5.0
        bbox = (
            float(mins[0] - margin), float(mins[1] - margin), float(mins[2] - margin),
            float(maxs[0] + margin), float(maxs[1] + margin), float(maxs[2] + margin),
        )

        pts_list = [tuple(p) for p in pts.tolist()]
        # Pin a single voxel resolution for every record. 3DTeethLand clouds
        # are all sparse (~40-115 landmarks), which straddles the adaptive
        # grid's n=100 boundary; without a fixed grid, coverage jumps ~3.3x
        # purely from the GRID=5 -> GRID=10 switch and values become
        # non-comparable across records.
        score = coverage_from_point_cloud(
            surface_region_id=rec.record_id,
            points=pts_list,
            region_bbox=bbox,
            grid=args.grid,
        )

        coverage_fractions.append(score.coverage_fraction)
        results.append({
            "record_id": rec.record_id,
            "arch": "upper" if "_upper" in rec.record_id else "lower",
            "n_landmarks": int(pts.shape[0]),
            "coverage_fraction": float(score.coverage_fraction),
            "occupied_voxels": score.stable_pixels,
            "total_voxels": score.total_pixels,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(records):
            mean_cov = float(np.mean(coverage_fractions)) if coverage_fractions else 0.0
            print(f"  [{i+1}/{len(records)}] {rec.record_id}: "
                  f"landmarks={pts.shape[0]}  coverage={score.coverage_fraction:.3f}  "
                  f"(running mean={mean_cov:.3f})")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "teethland_coverage.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"\nResults written to: {results_path}")
    print(f"Records processed:  {len(results)}")
    if coverage_fractions:
        arr = np.array(coverage_fractions)
        print(f"Coverage fraction:  mean={arr.mean():.3f}  std={arr.std():.3f}  min={arr.min():.3f}  max={arr.max():.3f}")
    if landmark_counts_by_class:
        print(f"\nLandmark class distribution:")
        for cls, count in sorted(landmark_counts_by_class.items(), key=lambda x: -x[1]):
            print(f"  {cls:20s}: {count}")


if __name__ == "__main__":
    main()
