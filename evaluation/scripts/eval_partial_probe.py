#!/usr/bin/env python3
"""Probe (#1): can robust scoring recover partial-overlap identity, training-free?

Rigid GICP collapses under tooth loss (keep-0.5 Rank-1 ~0.23) partly because the MEAN
query->gallery distance is inflated by crop-boundary points that land in anatomical gaps. Align
each 50%-cropped query to each gallery ONCE (the expensive step), then score it several ways --
mean (the baseline), trimmed-70%, median, and inlier-fitness -- and report which best separates
genuine from impostor. One alignment, many scores: a cheap, decisive test of where the
partial-overlap signal lives before committing to a learned correspondence model. Reuses the
exact eval_id3d augment/sample/align path so the `mean` column reproduces the baseline. Writes
partial_probe.json. Run from the shared eval dir (data/poseidon3d present).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths
from eval_id3d import NPTS, VX, augment, sample

from toothprint.identity import align_rigid

DATA = str(paths.POSEIDON3D)
OUT = Path(__file__).resolve().parents[1] / "results" / "partial_probe.json"
N, KEEP, TIGHT = 30, 0.5, 0.5            # 50% tooth loss; inlier threshold (mm) for fitness
SCORES = ["mean(baseline)", "trim70", "median", "1-fitness"]

_GAL = None


def _init(gal):
    global _GAL
    _GAL = gal


def _score(aligned, gal_pcd):
    d = np.asarray(o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(aligned)).compute_point_cloud_distance(gal_pcd))
    ds = np.sort(d)
    return (d.mean(), ds[:int(0.7 * len(ds))].mean(), float(np.median(d)), 1.0 - float((d < TIGHT).mean()))


def _row(task):
    i, = task
    q = augment(_GAL[i], 1000 + i, 0.06, 0.3, KEEP)
    out = []
    for g in _GAL:
        aligned, _ = align_rigid(q, g, VX)
        out.append(_score(aligned, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(g))))
    return i, np.array(out)                                   # (N, 4)


def rank1(M):
    return float(np.mean([np.argmin(M[i]) == i for i in range(len(M))]))


def main():
    meshes = sorted(Path(DATA).glob("*/*.stl"))[:N]
    raw = [r for r in (sample(m, NPTS) for m in meshes) if r is not None]
    workers = max(1, min(16, (mp.cpu_count() or 2) - 2))
    print(f"partial-overlap probe N={len(raw)} keep={KEEP}, {workers} workers ...", flush=True)
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(raw,)) as pool:
        rows = pool.map(_row, [(i,) for i in range(len(raw))])
    cube = np.zeros((len(raw), len(raw), 4))
    for i, r in rows:
        cube[i] = r
    res = {"n": len(raw), "keep": KEEP, "prior_baselines": {"gicp_mean": 0.23, "embedding": 0.52},
           "rank1": {SCORES[k]: round(rank1(cube[:, :, k]), 3) for k in range(4)}}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    for k in range(4):
        print(f"  {SCORES[k]:16s} Rank-1 {res['rank1'][SCORES[k]]:.3f}", flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
