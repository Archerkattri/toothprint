#!/usr/bin/env python3
"""Thorough 3D dental-identification evaluation with ablations.

Closed-set identification over real Poseidon3D arches. Genuine queries are
synthesised re-scans (rigid reposition + sensor noise + partial coverage), since
the dataset has one scan per subject. Reports Rank-1/Rank-5, genuine/impostor
separation (d'), ROC-AUC and EER, and ablates over sensor noise, tooth loss
(partial coverage), and scan resolution.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

DATA = "data/poseidon3d/extracted/data"
OUT = Path("/home/krishi/personal-projects/toothprint/evaluation/results/id3d.json")
RANSAC_ITERS = 60000


def sample(path, n):
    m = o3d.io.read_triangle_mesh(str(path))
    return np.asarray(m.sample_points_uniformly(number_of_points=n).points) if len(m.triangles) else None


def pcd(pts, vx):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(vx)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vx * 2, max_nn=30))
    return p


def fpfh(p, vx):
    return o3d.pipelines.registration.compute_fpfh_feature(
        p, o3d.geometry.KDTreeSearchParamHybrid(radius=vx * 5, max_nn=100))


def reg(q, qf, g, gf, vx):
    d = vx * 1.5
    c = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        q, g, qf, gf, True, d, o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(d)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(RANSAC_ITERS, 0.999))
    f = o3d.pipelines.registration.registration_icp(
        q, g, d, c.transformation, o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return float(f.inlier_rmse) if f.fitness > 0 else 1e6


def make_query(pts, rng, noise, keep):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax); a = rng.uniform(-0.3, 0.3)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)
    p = pts @ R.T + rng.uniform(-5, 5, 3)
    if keep < 1.0:
        c = p.mean(0); n = rng.normal(size=3); n /= np.linalg.norm(n)
        p = p[(p - c) @ n >= np.quantile((p - c) @ n, 1 - keep)]
    idx = rng.choice(len(p), min(len(p), len(pts)), replace=False)
    return p[idx] + rng.normal(0, noise, (len(idx), 3))


def rmse_matrix(meshes, noise, keep, vx, npts, seed=0):
    rng = np.random.default_rng(seed)
    raw = [sample(m, npts) for m in meshes]
    keep_idx = [i for i, r in enumerate(raw) if r is not None]
    raw = [raw[i] for i in keep_idx]
    labels = [meshes[i].parent.name + "_" + meshes[i].stem.split("_")[-1] for i in keep_idx]
    gal = [(pcd(r, vx), fpfh(pcd(r, vx), vx)) for r in raw]
    M = np.zeros((len(raw), len(raw)))
    for i, r in enumerate(raw):
        qp = pcd(make_query(r, rng, noise, keep), vx); qf = fpfh(qp, vx)
        for j, (gp, gf) in enumerate(gal):
            M[i, j] = reg(qp, qf, gp, gf, vx)
    return M, labels


def metrics(M, labels):
    n = len(labels)
    gen, imp = [], []
    r1 = r5 = 0
    for i in range(n):
        order = np.argsort(M[i])
        if labels[order[0]] == labels[i]:
            r1 += 1
        if labels[i] in [labels[k] for k in order[:5]]:
            r5 += 1
        for j in range(n):
            (gen if i == j else imp).append(M[i, j])
    gen, imp = np.array(gen), np.array(imp)
    fail_rate = float((imp >= 1e5).mean())  # impostor pairs with no usable registration
    # ROC-AUC + EER use raw values (failed registrations are correctly large).
    auc = float(np.mean(gen[:, None] < imp[None, :])) if gen.size and imp.size else float("nan")
    thr = np.unique(np.concatenate([gen, np.clip(imp, 0, 2.0)]))
    far = np.array([(imp < t).mean() for t in thr]); frr = np.array([(gen > t).mean() for t in thr])
    eer = float(np.min(np.maximum(far, frr))) if thr.size else float("nan")
    # d' on values capped at 2 mm so the 1e6 "no-match" sentinel doesn't dominate.
    impc = np.clip(imp, 0, 2.0)
    dprime = float(abs(gen.mean() - impc.mean()) / np.sqrt((gen.var() + impc.var()) / 2)) if gen.size else 0.0
    return {"n": n, "rank1": r1 / n, "rank5": r5 / n, "auc": auc, "eer": eer, "dprime": dprime,
            "registration_fail_rate": fail_rate,
            "genuine_mean": float(gen.mean()), "genuine_max": float(gen.max()),
            "impostor_mean": float(impc.mean()), "impostor_min": float(imp.min()),
            "genuine": gen.tolist(), "impostor_round": [round(float(x), 4) for x in impc[:2000]]}


def main():
    meshes = sorted(Path(DATA).glob("*/*.stl"))
    N_MAIN, N_ABL = 50, 30
    res = {"dataset": "poseidon3d", "n_meshes_total": len(meshes), "ablations": {}}
    t0 = time.time()

    print(f"[main] full identification, N={N_MAIN}, default settings ...", flush=True)
    M, labels = rmse_matrix(meshes[:N_MAIN], noise=0.05, keep=0.75, vx=0.5, npts=3000)
    res["main"] = metrics(M, labels)
    print(f"  rank1={res['main']['rank1']:.3f} d'={res['main']['dprime']:.1f} "
          f"eer={res['main']['eer']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    abl = meshes[:N_ABL]
    for noise in [0.0, 0.1, 0.2, 0.4]:
        M, l = rmse_matrix(abl, noise=noise, keep=0.75, vx=0.5, npts=3000)
        res["ablations"][f"noise_{noise}"] = metrics(M, l)
        print(f"  noise {noise}mm: rank1={res['ablations'][f'noise_{noise}']['rank1']:.3f}", flush=True)
    for keep in [1.0, 0.5, 0.3, 0.2]:
        M, l = rmse_matrix(abl, noise=0.05, keep=keep, vx=0.5, npts=3000)
        res["ablations"][f"keep_{keep}"] = metrics(M, l)
        print(f"  tooth-coverage {keep}: rank1={res['ablations'][f'keep_{keep}']['rank1']:.3f}", flush=True)
    for vx in [0.3, 0.8, 1.2]:
        M, l = rmse_matrix(abl, noise=0.05, keep=0.75, vx=vx, npts=3000)
        res["ablations"][f"voxel_{vx}"] = metrics(M, l)
        print(f"  voxel {vx}mm: rank1={res['ablations'][f'voxel_{vx}']['rank1']:.3f}", flush=True)

    res["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(res, indent=1))
    print(f"saved {OUT}  ({res['seconds']:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
