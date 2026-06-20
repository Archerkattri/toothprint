#!/usr/bin/env python3
"""End-to-end REAL reconstruction error on Poseidon3D dental meshes.

Connects the full chain on real data with NO synthetic fallback:

    real STL mesh
      -> render_5_views (open3d offscreen renderer)   # 5 protocol photos
      -> reconstruct_point_cloud (VGGT or DUSt3R; NO fallback — raises on failure)
      -> surface_error_mm vs the mesh's own sampled point cloud (ground truth)

Every number printed is computed from the real mesh geometry. The reconstruction
backend (vggt or dust3r) runs strictly: there is no crude CPU fallback, so if it
cannot run the script fails loudly and the real cause is fixed.

Status: the vggt/dust3r backends are implemented; pending a GPU run (they need
a GPU, e.g. the 2x RTX 5090 target hardware).

Usage:
    # Metric default (needs a GPU):
    python scripts/run_dmc_reconstruction_real.py \
        --data data/poseidon3d/extracted/data \
        --output outputs/dmc_reconstruction_real \
        --backend vggt --limit 5 --resolution 256 --n-gt-points 5000 --seed 0

"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from dentalmapcert.dataset_loaders import Poseidon3DLoader, load_poseidon3d_points
from dentalmapcert.meshing import poisson_surface_reconstruction
from dentalmapcert.reconstruction import BACKENDS, reconstruct_point_cloud
from dentalmapcert.render import render_5_views, rendered_view_to_pil
from dentalmapcert.surface_error import surface_error_mm


def _render_and_save(mesh_path: Path, out_dir: Path, resolution: int) -> list[Path]:
    """Render the 5 protocol views of *mesh_path* and save them as PNGs."""
    views = render_5_views(str(mesh_path), resolution=resolution)
    paths: list[Path] = []
    for view_name, view in views.items():
        img = rendered_view_to_pil(view)
        p = out_dir / f"{view_name}.png"
        img.save(p)
        paths.append(p)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Real render->reconstruct->surface_error on Poseidon3D")
    parser.add_argument("--data", default="data/poseidon3d/extracted/data",
                        help="Root of extracted Poseidon3D data/ directory")
    parser.add_argument("--output", default="outputs/dmc_reconstruction_real")
    parser.add_argument("--limit", type=int, default=5,
                        help="Max number of meshes to process")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Render resolution per view")
    parser.add_argument("--n-gt-points", type=int, default=5000,
                        help="Points sampled from the GT mesh for the comparison")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for reproducible GT mesh sampling")
    parser.add_argument("--backend", choices=BACKENDS, default="vggt",
                        help="Metric neural reconstruction backend (vggt or dust3r). "
                             "Runs strictly — there is no crude CPU fallback; it raises "
                             "if the backend cannot run.")
    parser.add_argument("--mesh-refine", action="store_true",
                        help="Refine the reconstruction with screened-Poisson surface "
                             "reconstruction (denoises sub-mm..~1mm error by ~30-43%%; "
                             "past ~2mm it over-smooths and HURTS — see meshing.py)")
    args = parser.parse_args()

    root = Path(args.data)
    if not root.exists():
        print(f"ERROR: Poseidon3D data not found at {root}", file=sys.stderr)
        sys.exit(1)

    loader = Poseidon3DLoader(str(root))
    records = list(loader.records())
    if not records:
        print(f"ERROR: no Poseidon3D records under {root}", file=sys.stderr)
        sys.exit(1)
    records = records[: args.limit]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    chamfers_mm = []
    print(f"Running REAL render->reconstruct->surface_error on {len(records)} Poseidon3D meshes\n")
    for i, rec in enumerate(records):
        mesh_path = Path(rec.mesh_path)
        if not mesh_path.exists():
            print(f"  [{i+1}/{len(records)}] {rec.record_id}: mesh missing, skipped")
            continue

        # Ground-truth point cloud sampled from the real mesh.
        gt_points = load_poseidon3d_points(rec, n_points=args.n_gt_points, seed=args.seed)
        if gt_points.shape[0] < 3:
            print(f"  [{i+1}/{len(records)}] {rec.record_id}: empty mesh, skipped")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            image_paths = _render_and_save(mesh_path, Path(tmp), args.resolution)
            recon_points, _conf = reconstruct_point_cloud(image_paths, backend=args.backend)

        if recon_points.shape[0] < 3:
            print(f"  [{i+1}/{len(records)}] {rec.record_id}: reconstruction empty, skipped")
            continue

        if args.mesh_refine:
            recon_points = poisson_surface_reconstruction(recon_points)

        # Neural backends (VGGT/DUSt3R) recover geometry up to an unknown global
        # scale, so align with a similarity transform (scale + rigid) to measure
        # shape fidelity rather than the arbitrary reconstruction scale.
        err = surface_error_mm(recon_points, gt_points, run_icp=True, estimate_scale=True)
        chamfers_mm.append(err.chamfer_mm)
        results.append({
            "record_id": rec.record_id,
            "n_views": len(image_paths),
            "n_gt_points": int(gt_points.shape[0]),
            "n_recon_points": int(recon_points.shape[0]),
            "chamfer_mm": float(err.chamfer_mm),
            "point_to_surface_mm": float(err.point_to_surface_mm),
            "hausdorff_mm": float(err.hausdorff_mm),
            "icp_iterations": int(err.icp_iterations),
            "icp_converged": bool(err.icp_converged),
        })
        print(f"  [{i+1}/{len(records)}] {rec.record_id}: "
              f"chamfer={err.chamfer_mm:.3f} mm  p2s={err.point_to_surface_mm:.3f} mm  "
              f"hausdorff={err.hausdorff_mm:.3f} mm  ({recon_points.shape[0]} recon pts)")

    payload = {
        "dataset": "poseidon3d",
        "n_meshes": len(results),
        "resolution": args.resolution,
        "n_gt_points": args.n_gt_points,
        "seed": args.seed,
        "backend": args.backend,
        "synthetic": False,
        "records": results,
    }
    if chamfers_mm:
        arr = np.array(chamfers_mm, dtype=float)
        payload["chamfer_mm_mean"] = float(arr.mean())
        payload["chamfer_mm_std"] = float(arr.std())
        payload["chamfer_mm_min"] = float(arr.min())
        payload["chamfer_mm_max"] = float(arr.max())

    out_path = out_dir / "reconstruction_error.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\nResults written to: {out_path}")
    if chamfers_mm:
        print(f"REAL Chamfer error: mean={payload['chamfer_mm_mean']:.3f} mm  "
              f"std={payload['chamfer_mm_std']:.3f} mm  "
              f"min={payload['chamfer_mm_min']:.3f}  max={payload['chamfer_mm_max']:.3f} "
              f"(n={len(results)} real meshes)")
    else:
        print("No reconstructions succeeded.")


if __name__ == "__main__":
    main()
