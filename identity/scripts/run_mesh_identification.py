#!/usr/bin/env python3
"""Validate 3D dental identification on real Poseidon3D intraoral-scan meshes.

Builds a gallery from N subjects' arch meshes, then for each subject synthesises a
realistic "second scan" query (random rigid repositioning + scan noise + partial
coverage + independent resampling) and identifies it against the whole gallery by
the smallest registration RMSE. Reports Rank-1 accuracy and the genuine/impostor
RMSE separation — the recognition of a person by their teeth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # toothid on path
from toothid.mesh_id import enroll, identification_metrics, register_rmse


def _sample_mesh(path, n, seed=0):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(path))
    if len(m.triangles) == 0:
        return None
    pcd = m.sample_points_uniformly(number_of_points=n)
    return np.asarray(pcd.points)


def _make_query(points, rng, noise_mm, keep_frac):
    """Simulate a re-scan: rigid reposition + noise + partial coverage + resample."""
    # random rotation
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    ang = rng.uniform(-0.3, 0.3)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    t = rng.uniform(-5, 5, 3)
    pts = points @ R.T + t
    # partial coverage: drop points beyond a random plane (simulates a partial scan)
    c = pts.mean(axis=0)
    nrm = rng.normal(size=3); nrm /= np.linalg.norm(nrm)
    d = (pts - c) @ nrm
    thresh = np.quantile(d, 1.0 - keep_frac)
    pts = pts[d >= thresh]
    # resample (subset) + scan noise
    idx = rng.choice(len(pts), size=min(len(pts), len(points)), replace=False)
    pts = pts[idx] + rng.normal(0, noise_mm, (len(idx), 3))
    return pts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../surface/data/poseidon3d/extracted/data")
    p.add_argument("--output", default="outputs/mesh_identification")
    p.add_argument("--n-subjects", type=int, default=18)
    p.add_argument("--n-points", type=int, default=4000)
    p.add_argument("--voxel", type=float, default=0.5)
    p.add_argument("--noise-mm", type=float, default=0.05)
    p.add_argument("--keep-frac", type=float, default=0.75)
    args = p.parse_args()

    root = Path(args.data)
    if not root.exists():
        print(f"ERROR: Poseidon3D meshes not found at {root}", file=sys.stderr)
        return 1
    meshes = sorted(root.glob("*/*.stl"))[: args.n_subjects]
    if len(meshes) < 2:
        print(f"ERROR: need >=2 meshes, found {len(meshes)}", file=sys.stderr)
        return 1
    labels = [m.parent.name + "_" + m.stem.split("_")[-1] for m in meshes]
    print(f"Enrolling {len(meshes)} dental arches into the gallery...")

    rng = np.random.default_rng(0)
    gallery = []
    gpts = []
    for m in meshes:
        pts = _sample_mesh(m, args.n_points, seed=0)
        gpts.append(pts)
        gallery.append(enroll(pts, args.voxel))

    print("Synthesising re-scan queries and identifying each against the gallery...")
    rmse = np.zeros((len(meshes), len(meshes)))
    for i, pts in enumerate(gpts):
        qpts = _make_query(pts, rng, args.noise_mm, args.keep_frac)
        q_pcd, q_fpfh = enroll(qpts, args.voxel)
        for j, (g_pcd, g_fpfh) in enumerate(gallery):
            r, fit = register_rmse(q_pcd, q_fpfh, g_pcd, g_fpfh, args.voxel)
            # No usable alignment (fitness 0) -> treat as maximally dissimilar.
            rmse[i, j] = r if fit > 0 else 1e6
        best = int(np.argmin(rmse[i]))
        tag = "OK " if labels[best] == labels[i] else "MISS"
        print(f"  [{i+1}/{len(meshes)}] query {labels[i]:>22} -> {labels[best]:>22}  "
              f"rmse={rmse[i, best]:.3f}  {tag}")

    metrics = identification_metrics(rmse, labels, labels)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(
        {**metrics, "rmse": rmse.tolist(), "labels": labels}, indent=2))

    print(f"\n=== 3D dental identification (real Poseidon3D arches) ===")
    print(f"  Rank-1 accuracy:        {metrics['rank1_accuracy']:.3f} "
          f"({metrics['n_query']} queries vs {metrics['n_gallery']} gallery)")
    print(f"  genuine RMSE mean/max:  {metrics['genuine_rmse_mean']:.3f} / {metrics['genuine_rmse_max']:.3f} mm")
    print(f"  impostor RMSE mean/min: {metrics['impostor_rmse_mean']:.3f} / {metrics['impostor_rmse_min']:.3f} mm")
    print(f"  decidability d':        {metrics['decidability_dprime']:.2f}")
    print(f"\n  Metrics: {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
