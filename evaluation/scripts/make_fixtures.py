#!/usr/bin/env python3
"""Generate tiny synthetic dental-arch fixtures so the 3D pipeline runs without off-machine data.

Each 'subject' is a parametric horseshoe tube mesh with a per-subject tooth-bump pattern, so
re-sampling one arch matches itself (genuine) while different subjects differ (impostor) — enough
for the identity / correspondence pipeline to run end-to-end and produce sensible *toy* numbers.
This is NOT the benchmark; it exercises the code paths. Writes (committed, tiny)
evaluation/fixtures/arches/<id>/<id>_upper.{stl,obj}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "arches"
N_SUBJ, N_T, N_RING = 8, 80, 12


def arch_mesh(seed):
    rng = np.random.default_rng(seed)
    R, n_teeth, amp, phase = 1.0 + 0.1 * rng.standard_normal(), int(rng.integers(10, 16)), 0.04 + 0.02 * rng.random(), rng.uniform(0, 2 * np.pi)
    tsize, bump = 0.12 + 0.03 * rng.random(), rng.uniform(0.8, 1.2, 16)        # per-tooth size = the individual signature
    t = np.linspace(0.15 * np.pi, 0.85 * np.pi, N_T)
    cx, cy = R * np.cos(t), R * 1.3 * np.sin(t)
    verts, tris = [], []
    for i in range(N_T):
        ti = int((t[i] - t[0]) / (t[-1] - t[0]) * n_teeth) % 16
        rad = tsize * (1 + amp * np.sin(n_teeth * t[i] + phase)) * bump[ti]
        nrm = np.array([np.cos(t[i]), 1.3 * np.sin(t[i]), 0.0]); nrm /= np.linalg.norm(nrm)
        for j in range(N_RING):
            a = 2 * np.pi * j / N_RING
            verts.append(np.array([cx[i], cy[i], 0.0]) + rad * (np.cos(a) * nrm + np.sin(a) * np.array([0, 0, 1.0])))
    for i in range(N_T - 1):
        for j in range(N_RING):
            a, b = i * N_RING + j, i * N_RING + (j + 1) % N_RING
            c, d = (i + 1) * N_RING + j, (i + 1) * N_RING + (j + 1) % N_RING
            tris += [[a, b, c], [b, d, c]]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(verts)), o3d.utility.Vector3iVector(np.asarray(tris)))
    m.compute_vertex_normals()
    return m


def main():
    for s in range(N_SUBJ):
        sid = f"{s + 1:06d}"; d = OUT / sid; d.mkdir(parents=True, exist_ok=True)
        m = arch_mesh(1000 + s)
        o3d.io.write_triangle_mesh(str(d / f"{sid}_upper.stl"), m)
        o3d.io.write_triangle_mesh(str(d / f"{sid}_upper.obj"), m)
    print(f"wrote {N_SUBJ} synthetic arches to {OUT}")


if __name__ == "__main__":
    main()
