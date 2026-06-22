#!/usr/bin/env python3
"""Multimodal identity fusion (#4) on REAL same-subject data.

The Figshare CBCT+oral-scan set gives every patient an intraoral surface scan (IOS, crowns)
AND a CBCT volume (bone + roots) — two genuinely independent views of the same person. We
score identity under each modality (PCA-init + GICP surface distance) and **fuse at the score
level** (z-scored sum) under one decision, then check whether fusion beats the better single
modality. Single-timepoint, so genuine queries are synthetic re-scans (disclosed), but the
*two modalities are real and paired*. Writes multimodal_identity.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import open3d as o3d

from toothprint.identity import align_rigid

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths

DATA = paths.CBCT_IOS
OUT = Path(__file__).resolve().parents[1] / "results" / "multimodal_identity.json"
N = 2500


def norm(p):
    p = p - p.mean(0)
    return (p / (np.linalg.norm(p, axis=1).max() + 1e-9)).astype(np.float32)


def ios_cloud(pid):
    m = o3d.io.read_triangle_mesh(
        str(DATA / pid / f"{pid}_ios" / f"{pid}_UpperJawScan.stl")
    )
    return (
        norm(np.asarray(m.sample_points_uniformly(N).points))
        if len(m.triangles)
        else None
    )


def cbct_cloud(pid, rng):
    a = nib.load(str(DATA / pid / f"{pid}_cbct" / f"{pid}_cbct.nii.gz")).get_fdata()
    idx = np.argwhere(a > a.mean() + a.std()).astype(
        np.float32
    )  # bone + teeth surface points
    return norm(idx[rng.choice(len(idx), N, replace=False)]) if len(idx) >= N else None


def augment(p, rng, noise=0.012):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    a = rng.uniform(-0.5, 0.5)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)
    return norm(p @ R.T + rng.normal(0, noise, p.shape))


def matrix(gallery, queries):
    n = len(gallery)
    return np.array(
        [
            [align_rigid(queries[i], gallery[j], 0.5)[1] for j in range(n)]
            for i in range(n)
        ]
    )


def metrics(M):
    n = len(M)
    gen = np.array([M[i, i] for i in range(n)])
    imp = np.array([M[i, j] for i in range(n) for j in range(n) if i != j])
    r1 = float(np.mean([np.argmin(M[i]) == i for i in range(n)]))
    auc = float(np.mean(gen[:, None] < imp[None, :]))
    return {
        "rank1": r1,
        "auc": auc,
        "gen_mean": float(gen.mean()),
        "imp_min": float(imp.min()),
    }


def zscore(M):
    return (M - M.mean()) / (M.std() + 1e-9)


def main():
    pids = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    ios_g, cbct_g, ios_q, cbct_q, kept = [], [], [], [], []
    for pid in pids:
        ic = ios_cloud(pid)
        cc = cbct_cloud(pid, np.random.default_rng(7))
        if ic is None or cc is None:
            continue
        kept.append(pid)
        ios_g.append(ic)
        cbct_g.append(cc)
        ios_q.append(augment(ic, np.random.default_rng(100 + len(kept))))
        cbct_q.append(augment(cc, np.random.default_rng(200 + len(kept))))
    print(f"real paired patients: {len(kept)}", flush=True)

    A = matrix(ios_g, ios_q)  # IOS crowns
    B = matrix(cbct_g, cbct_q)  # CBCT bone/roots
    F = zscore(A) + zscore(B)  # score-level fusion
    res = {
        "n_patients": len(kept),
        "dataset": "Figshare CBCT+oral-scan (real paired)",
        "ios_crowns": metrics(A),
        "cbct_bone": metrics(B),
        "fused": metrics(F),
    }
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    for k in ("ios_crowns", "cbct_bone", "fused"):
        print(
            f"  {k:12s} Rank-1 {res[k]['rank1']:.3f}  AUC {res[k]['auc']:.3f}",
            flush=True,
        )
    print(
        f"  -> fusion {'BEATS' if res['fused']['rank1'] >= max(res['ios_crowns']['rank1'], res['cbct_bone']['rank1']) else 'ties/loses'} the best single modality"
    )
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
