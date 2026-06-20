#!/usr/bin/env python3
"""DentalMapCert coverage evaluation on Poseidon3D 3D dental meshes.

Loads Poseidon3D STL meshes, samples point clouds from them, and evaluates
the coverage_from_point_cloud function with realistic 3D dental data.

Each case is evaluated on both available arches (mandible/maxilla). The STL
mesh is sampled to produce a dense point cloud, then coverage_from_point_cloud
is called with a bounding-box that encloses each arch.

Poseidon3D: 200 IOS dental scan cases, STL meshes + tooth landmark markers,
CC-BY-4.0, Zenodo 15608906.

Usage:
    python scripts/run_dmc_poseidon3d.py \\
        --data data/poseidon3d/extracted/data \\
        --output outputs/dmc_poseidon3d \\
        [--n-points 5000] [--limit 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from dentalmapcert.coverage import coverage_from_point_cloud
from dentalmapcert.dataset_loaders import Poseidon3DLoader, load_poseidon3d_points
from dentalmapcert.surface_error import surface_error_mm


def _arch_bbox(points: np.ndarray, margin: float = 2.0) -> tuple:
    """Return axis-aligned bounding box of a point cloud with a margin."""
    if points.shape[0] == 0:
        return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return (
        float(mins[0] - margin), float(mins[1] - margin), float(mins[2] - margin),
        float(maxs[0] + margin), float(maxs[1] + margin), float(maxs[2] + margin),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DMC coverage evaluation on Poseidon3D")
    parser.add_argument("--data", default="data/poseidon3d/extracted/data",
                        help="Root of extracted Poseidon3D data/ directory (contains metadata.json)")
    parser.add_argument("--output", default="outputs/dmc_poseidon3d",
                        help="Output directory for results JSON")
    parser.add_argument("--n-points", type=int, default=5000,
                        help="Number of points to sample from each STL mesh (default=5000)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of records to process (None = all)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for reproducible mesh surface sampling (default=0)")
    args = parser.parse_args()

    root = Path(args.data)
    if not root.exists():
        print(f"ERROR: Poseidon3D data not found at {root}", file=sys.stderr)
        sys.exit(1)

    try:
        import open3d  # noqa: F401
    except ImportError:
        print("ERROR: open3d is required for STL point cloud sampling.", file=sys.stderr)
        print("Install with: pip install open3d", file=sys.stderr)
        sys.exit(1)

    loader = Poseidon3DLoader(str(root))
    records = list(loader.records())
    print(f"Found {len(records)} Poseidon3D records (mandible + maxilla arches)")

    if args.limit:
        records = records[: args.limit]
        print(f"Processing first {args.limit} records")

    results = []
    coverage_fractions = []
    chamfer_vals = []

    for i, rec in enumerate(records):
        arch = "mandible" if "mandible" in rec.notes else "maxilla"
        case_id = rec.record_id.replace("poseidon3d_", "").replace(f"_{arch}", "")

        pts = load_poseidon3d_points(rec, n_points=args.n_points, seed=args.seed)
        if pts.shape[0] == 0:
            print(f"  [{i+1}/{len(records)}] {rec.record_id}: skipped (no points sampled)")
            continue

        # Self-consistency check: surface_error between two samples from the same mesh
        # (a different seed + point count) gives a baseline reconstruction noise estimate.
        pts_repeat = load_poseidon3d_points(rec, n_points=args.n_points // 2, seed=args.seed + 1)
        if pts_repeat.shape[0] > 0 and pts.shape[0] > 0:
            try:
                err = surface_error_mm(pts, pts_repeat, run_icp=False)
                chamfer_vals.append(err.chamfer_mm)
            except Exception:
                pass

        bbox = _arch_bbox(pts)
        pts_list = [tuple(p) for p in pts.tolist()]
        score = coverage_from_point_cloud(
            surface_region_id=rec.record_id,
            points=pts_list,
            region_bbox=bbox,
        )

        coverage_fractions.append(score.coverage_fraction)
        results.append({
            "record_id": rec.record_id,
            "case_id": case_id,
            "arch": arch,
            "n_points_sampled": int(pts.shape[0]),
            "coverage_fraction": float(score.coverage_fraction),
            "occupied_voxels": score.stable_pixels,
            "total_voxels": score.total_pixels,
            "method": score.method,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            mean_cov = float(np.mean(coverage_fractions)) if coverage_fractions else 0.0
            print(f"  [{i+1}/{len(records)}] {rec.record_id}: coverage={score.coverage_fraction:.3f}  (running mean={mean_cov:.3f})")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "poseidon3d_coverage.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"\nResults written to: {results_path}")
    print(f"Records processed:  {len(results)}")
    if coverage_fractions:
        arr = np.array(coverage_fractions)
        print(f"Coverage fraction:  mean={arr.mean():.3f}  std={arr.std():.3f}  min={arr.min():.3f}  max={arr.max():.3f}")
    if chamfer_vals:
        arr_c = np.array(chamfer_vals)
        print(f"Self-consistency Chamfer: mean={arr_c.mean():.3f} std={arr_c.std():.3f} mm")


if __name__ == "__main__":
    main()
