#!/usr/bin/env python3
"""Explicit per-region (per-tooth-band) identity (#6) — the interpretable, forensic-chart view.

Whole-arch identity scores one mean surface distance; this instead aligns (PCA+GICP), then
scores each arch *region* (a band along the arch) separately and aggregates the **best-K**
regions — so a partial arch (missing teeth) is judged on the teeth it still has, and the
output says *which* regions matched (a digital dental chart). Compared head-to-head with the
whole-arch mean at full and partial coverage. Writes regional_identity.json.

Run from the shared eval dir (data/poseidon3d present).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_id3d import NPTS, VX, sample

from toothprint.identity import align_rigid

DATA = "data/poseidon3d/extracted/data"
OUT = Path(__file__).resolve().parents[1] / "results" / "regional_identity.json"
K, BEST_K, N = 12, 8, 45


def augment(p, rng, keep, noise=0.05):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax); a = rng.uniform(-0.3, 0.3)
    Kk = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(a) * Kk + (1 - np.cos(a)) * (Kk @ Kk)
    q = p @ R.T + rng.uniform(-5, 5, 3)
    if keep < 1.0:
        c = q.mean(0); n = rng.normal(size=3); n /= np.linalg.norm(n)
        q = q[(q - c) @ n >= np.quantile((q - c) @ n, 1 - keep)]
    return (q + rng.normal(0, noise, q.shape)).astype(np.float32)


def regional_dist(aligned, gal_pcd, gal_pts):
    c = gal_pts.mean(0); u = np.linalg.svd(gal_pts - c, full_matrices=False)[2][0]   # arch axis
    proj = (aligned - c) @ u
    edges = np.linspace(proj.min(), proj.max(), K + 1)
    d = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned)).compute_point_cloud_distance(gal_pcd))
    reg = [d[(proj >= edges[b]) & (proj < edges[b + 1])].mean()
           for b in range(K) if ((proj >= edges[b]) & (proj < edges[b + 1])).sum() > 5]
    return float(np.mean(sorted(reg)[:BEST_K])) if reg else 1e6


def rank1(M):
    return float(np.mean([np.argmin(M[i]) == i for i in range(len(M))]))


def main():
    meshes = sorted(Path(DATA).glob("*/*.stl"))[:N]
    pts = [p for p in (sample(m, NPTS) for m in meshes) if p is not None]
    pcds = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p)) for p in pts]
    n = len(pts)
    print(f"regional identity, N={n}, K={K} regions best-{BEST_K} ...", flush=True)
    res = {"n": n, "regions": K, "best_k": BEST_K, "keep_ablation": {}}
    for keep in [1.0, 0.5, 0.3]:
        Wh = np.zeros((n, n)); Rg = np.zeros((n, n))
        for i in range(n):
            q = augment(pts[i], np.random.default_rng(500 + i), keep)
            for j in range(n):
                aligned, md = align_rigid(q, pts[j], VX)
                Wh[i, j] = md
                Rg[i, j] = regional_dist(aligned, pcds[j], pts[j])
        res["keep_ablation"][str(keep)] = {"whole_arch_rank1": rank1(Wh), "regional_bestK_rank1": rank1(Rg)}
        print(f"  keep {keep}: whole-arch {rank1(Wh):.3f}  ->  regional best-{BEST_K} {rank1(Rg):.3f}", flush=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
