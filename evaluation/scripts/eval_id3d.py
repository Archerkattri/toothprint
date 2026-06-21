#!/usr/bin/env python3
"""Thorough 3D dental-identification evaluation over real Poseidon3D arches.

Uses toothprint's own identity method — PCA principal-axis init + multi-scale
Generalized-ICP, scored by post-alignment mean surface distance
(:func:`toothprint.identity.align_rigid`). Genuine queries are synthesised re-scans
(rigid reposition + sensor noise + partial coverage), since the dataset has one scan
per subject. The full N=200 x N=200 matrix is the biometric; ablations (a subset)
sweep sensor noise, tooth loss (partial coverage), and scan resolution. Reports
Rank-1/Rank-5, genuine/impostor separation (d'), ROC-AUC, and EER.

Run from the shared evaluation working dir (data/poseidon3d present). CPU-bound and
parallel; ~tens of minutes at N=200.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from toothprint.identity import align_rigid

DATA = "data/poseidon3d/extracted/data"
OUT = Path(__file__).resolve().parents[1] / "results" / "id3d.json"
# Core augmentation: a genuine re-scan is the arch repositioned + sensor noise, full
# coverage. Matches the parameters behind the headline N=200 result.
NOISE, ROT_AMP, KEEP, NPTS, VX = 0.06, 0.3, 1.0, 3500, 0.5

_GAL: list | None = None       # gallery point sets, shared read-only across workers


def sample(path, n):
    m = o3d.io.read_triangle_mesh(str(path))
    return np.asarray(m.sample_points_uniformly(number_of_points=n).points) if len(m.triangles) else None


def _rot(rng, amp):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax); a = rng.uniform(-amp, amp)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def augment(pts, seed, noise, rot_amp, keep):
    rng = np.random.default_rng(seed)
    p = pts @ _rot(rng, rot_amp).T + rng.uniform(-5, 5, 3)
    if keep < 1.0:                                  # partial-overlap crop (tooth loss)
        c = p.mean(0); n = rng.normal(size=3); n /= np.linalg.norm(n)
        proj = (p - c) @ n
        p = p[proj >= np.quantile(proj, 1 - keep)]
    return p + rng.normal(0, noise, p.shape)


def _init(gal):
    global _GAL
    _GAL = gal


def _row(task):
    i, noise, rot_amp, keep, vx = task
    q = augment(_GAL[i], 1000 + i, noise, rot_amp, keep)
    return i, np.array([align_rigid(q, g, vx)[1] for g in _GAL])


def build_matrix(meshes, noise, rot_amp, keep, vx, npts, workers):
    raw = [sample(m, npts) for m in meshes]
    idx = [i for i, r in enumerate(raw) if r is not None]
    raw = [raw[i] for i in idx]
    labels = [meshes[i].parent.name + "_" + meshes[i].stem.split("_")[-1] for i in idx]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(raw,)) as pool:
        rows = pool.map(_row, [(i, noise, rot_amp, keep, vx) for i in range(len(raw))])
    M = np.zeros((len(raw), len(raw)))
    for i, row in rows:
        M[i] = row
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
    fail_rate = float((imp >= 1e5).mean())
    auc = float(np.mean(gen[:, None] < imp[None, :])) if gen.size and imp.size else float("nan")
    thr = np.unique(np.concatenate([gen, np.clip(imp, 0, 3.0)]))
    far = np.array([(imp < t).mean() for t in thr]); frr = np.array([(gen > t).mean() for t in thr])
    eer = float(np.min(np.maximum(far, frr))) if thr.size else float("nan")
    impc = np.clip(imp, 0, 3.0)
    dprime = float(abs(gen.mean() - impc.mean()) / np.sqrt((gen.var() + impc.var()) / 2)) if gen.size else 0.0
    return {"n": n, "rank1": r1 / n, "rank5": r5 / n, "auc": auc, "eer": eer, "dprime": dprime,
            "registration_fail_rate": fail_rate,
            "genuine_mean": float(gen.mean()), "genuine_max": float(gen.max()),
            "impostor_mean": float(impc.mean()), "impostor_min": float(imp.min()),
            "genuine": gen.tolist(), "impostor_round": [round(float(x), 4) for x in impc[:2000]]}


def main():
    meshes = sorted(Path(DATA).glob("*/*.stl"))
    workers = max(1, min(16, (os.cpu_count() or 2) - 2))
    N_MAIN, N_ABL = len(meshes), 30
    res = {"dataset": "poseidon3d", "n_meshes_total": len(meshes), "method": "PCA-init + GICP, point-to-point",
           "ablations": {}}
    t0 = time.time()

    # The headline N=200 matrix is committed (core_p2p.npy) and analysed by
    # analyze_identity.py; reuse it for the main panel so this grid matches the
    # headline exactly, and recompute only the (fast) robustness ablations live.
    mat_path = OUT.parent / "core_p2p.npy"
    print(f"[main] full identification, N={N_MAIN}, {workers} workers ...", flush=True)
    if mat_path.exists():
        M = np.load(mat_path); labels = [str(i) for i in range(len(M))]
        res["main_source"] = "committed core_p2p.npy (PCA+GICP point-to-point)"
    else:
        M, labels = build_matrix(meshes[:N_MAIN], NOISE, ROT_AMP, KEEP, VX, NPTS, workers)
        res["main_source"] = "recomputed"
    res["main"] = metrics(M, labels)
    print(f"  rank1={res['main']['rank1']:.3f} eer={res['main']['eer']:.3f} auc={res['main']['auc']:.3f} "
          f"d'={res['main']['dprime']:.2f} ({time.time()-t0:.0f}s)", flush=True)

    abl = meshes[:N_ABL]
    for noise in [0.0, 0.1, 0.2, 0.4]:
        M, l = build_matrix(abl, noise, ROT_AMP, KEEP, VX, NPTS, workers)
        res["ablations"][f"noise_{noise}"] = metrics(M, l)
        print(f"  noise {noise}mm: rank1={res['ablations'][f'noise_{noise}']['rank1']:.3f}", flush=True)
    for keep in [1.0, 0.5, 0.3, 0.2]:
        M, l = build_matrix(abl, NOISE, ROT_AMP, keep, VX, NPTS, workers)
        res["ablations"][f"keep_{keep}"] = metrics(M, l)
        print(f"  tooth-coverage {keep}: rank1={res['ablations'][f'keep_{keep}']['rank1']:.3f}", flush=True)
    for vx in [0.3, 0.8, 1.2]:
        M, l = build_matrix(abl, NOISE, ROT_AMP, KEEP, vx, NPTS, workers)
        res["ablations"][f"voxel_{vx}"] = metrics(M, l)
        print(f"  voxel {vx}mm: rank1={res['ablations'][f'voxel_{vx}']['rank1']:.3f}", flush=True)

    res["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(res, indent=1))
    print(f"saved {OUT}  ({res['seconds']:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
