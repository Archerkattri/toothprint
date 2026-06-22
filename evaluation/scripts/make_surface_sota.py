#!/usr/bin/env python3
"""Surface change vs the SOTA baseline (M3C2) — detecting a LOCALIZED lesion.

M3C2 (Lague 2013) is the geomorphology-standard cloud-to-cloud change distance:
the along-normal displacement averaged over a local neighbourhood. It is a strong
per-point distance, but to flag a small *localized* lesion you still have to reduce
its field to a scalar — and the naive whole-surface mean dilutes the lesion to nothing,
while its per-point max is dominated by reconstruction noise.

Ours = de-biased (noise-power-subtracted) displacement measured PER REGION, with the
max-region calibrated conformally. This benchmark sweeps reconstruction noise and
reports localized-change recall at a fixed false-change rate for: (A) naive global
mean, (B) M3C2 max, (C) ours. Writes docs/surface_sota.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from toothprint.surface.error import (assign_regions, noise_floor_sq,
                                      regional_displacements, surface_displacement)

DATA = "data/poseidon3d/extracted/data"
N = 6000
NREG = 32              # finer partition localizes a small lesion better
LESION_FRAC = 0.02     # a SMALL localized patch (~2% of the surface)
LESION_MM = 0.5        # how far the lesion moves (along normal)
K = 30                 # trials per condition
ALPHA = 0.05           # target false-change rate


def m3c2_max(t0, t1, normals, radius, tree):
    """M3C2-style localized indicator: max |along-normal, neighbourhood-averaged disp|."""
    v = t1 - t0
    out = np.zeros(len(t0))
    for i in range(0, len(t0), 4):           # stride 4 for speed; representative
        idx = tree.query_ball_point(t0[i], radius)
        out[i] = abs((v[idx] @ normals[i]).mean()) if idx else 0.0
    return float(out.max())


def main():
    mesh = o3d.io.read_triangle_mesh(str(sorted(Path(DATA).glob("*/*.stl"))[0]))
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(N)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
    t0 = np.asarray(pcd.points); nrm = np.asarray(pcd.normals)
    scale = np.linalg.norm(t0.max(0) - t0.min(0))
    radius = 0.03 * scale
    tree = cKDTree(t0)
    labels = assign_regions(t0, n_regions=NREG)

    # localized lesion mask: points within a ball of a random centre
    rng = np.random.default_rng(0)
    c = t0[rng.integers(len(t0))]
    order = np.argsort(((t0 - c) ** 2).sum(1))
    lesion = np.zeros(len(t0), bool); lesion[order[:int(LESION_FRAC * len(t0))]] = True

    sigmas = [0.1, 0.2, 0.3, 0.4, 0.6]
    recall = {"whole-surface average (de-biased)": [], "M3C2 max": [], "ours (regional + conformal)": []}
    for sig in sigmas:
        # calibrate each method's threshold on STABLE pairs (no lesion) at FPR=alpha
        def stable_scores():
            gm, m3, our = [], [], []
            floors = None
            cal_t1 = [t0 + rng.normal(0, sig, t0.shape) for _ in range(K)]
            floors = noise_floor_sq([(t0, x) for x in cal_t1])  # global noise power
            # per-region floors from stable pairs
            rfloors = np.array([noise_floor_sq([(t0[labels == r], x[labels == r]) for x in cal_t1])
                                for r in range(NREG)])
            for x in cal_t1:
                gm.append(surface_displacement(t0, x, noise_floor_sq=floors))   # de-biased whole-surface
                m3.append(m3c2_max(t0, x, nrm, radius, tree))
                our.append(float(regional_displacements(t0, x, labels, rfloors).max()))
            return np.array(gm), np.array(m3), np.array(our), floors, rfloors
        s_gm, s_m3, s_our, gfloor, rfloors = stable_scores()
        thr = {"gm": np.quantile(s_gm, 1 - ALPHA), "m3": np.quantile(s_m3, 1 - ALPHA),
               "our": np.quantile(s_our, 1 - ALPHA)}
        # recall on LESION pairs
        hit = {"gm": 0, "m3": 0, "our": 0}
        for _ in range(K):
            x = t0.copy()
            x[lesion] += LESION_MM * nrm[lesion]
            x += rng.normal(0, sig, x.shape)
            if surface_displacement(t0, x, noise_floor_sq=gfloor) > thr["gm"]: hit["gm"] += 1
            if m3c2_max(t0, x, nrm, radius, tree) > thr["m3"]: hit["m3"] += 1
            if regional_displacements(t0, x, labels, rfloors).max() > thr["our"]: hit["our"] += 1
        recall["whole-surface average (de-biased)"].append(hit["gm"] / K)
        recall["M3C2 max"].append(hit["m3"] / K)
        recall["ours (regional + conformal)"].append(hit["our"] / K)
        print(f"  sigma={sig}: global {hit['gm']/K:.2f}  M3C2 {hit['m3']/K:.2f}  ours {hit['our']/K:.2f}", flush=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = {"whole-surface average (de-biased)": "#9aa7ad", "M3C2 max": "#e0954a", "ours (regional + conformal)": "#2ca06b"}
    for k, v in recall.items():
        ax.plot(sigmas, v, "-o", lw=2.4, color=colors[k], label=k)
    ax.axhline(ALPHA, ls=":", color="#c0392b", lw=1.2)
    ax.text(sigmas[0], ALPHA + 0.02, f"false-change rate held at α={ALPHA}", color="#c0392b", fontsize=9)
    ax.set_xlabel("reconstruction noise σ (mm)"); ax.set_ylabel(f"localized-change recall @ {int(ALPHA*100)}% false-change")
    ax.set_ylim(-0.03, 1.05); ax.legend(fontsize=10, loc="center left")
    ax.set_title(f"Localized surface change vs the M3C2 baseline\n"
                 f"a {LESION_MM} mm lesion over {int(LESION_FRAC*100)}% of the arch, under reconstruction noise",
                 fontsize=12)
    fig.patch.set_facecolor("white"); fig.tight_layout()
    fig.savefig("docs/surface_sota.png", dpi=120); plt.close(fig)
    print("wrote docs/surface_sota.png")


if __name__ == "__main__":
    main()
