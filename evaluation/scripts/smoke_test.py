#!/usr/bin/env python3
"""End-to-end smoke test on committed synthetic fixtures — proves the identity pipeline runs
with NO off-machine data. The point is reproducibility, not the benchmark: it asserts the code
path works and produces sensible (toy) separation on the fixture arches.

    TOOTHPRINT_FIXTURES=1 python evaluation/scripts/smoke_test.py     # or: pytest evaluation/scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TOOTHPRINT_FIXTURES", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import open3d as o3d

import paths
from toothprint.identity import align_rigid


def test_identity_pipeline_runs_on_fixtures():
    meshes = sorted(paths.POSEIDON3D.glob("*/*.stl"))
    assert meshes, f"no fixtures at {paths.POSEIDON3D} — run make_fixtures.py first"
    pts = [np.asarray(o3d.io.read_triangle_mesh(str(m)).sample_points_uniformly(1500).points) for m in meshes]
    n = len(pts)
    M = np.zeros((n, n))
    for i in range(n):
        q = pts[i] + np.random.default_rng(i).normal(0, 0.003, pts[i].shape)      # genuine = re-sample + jitter
        for j in range(n):
            M[i, j] = align_rigid(q, pts[j], 0.02)[1]
    r1 = float(np.mean([np.argmin(M[i]) == i for i in range(n)]))
    print(f"fixture identity: Rank-1 {r1:.3f}  (n={n}, chance {1/n:.3f}) — align_rigid + surface-distance pipeline OK")
    assert r1 > 0.6, f"pipeline produced no separation on fixtures (Rank-1 {r1:.3f})"


if __name__ == "__main__":
    test_identity_pipeline_runs_on_fixtures()
    print("SMOKE OK")
