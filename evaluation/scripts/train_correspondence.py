#!/usr/bin/env python3
"""Train CorrNet: learned partial->whole point correspondence for partial-overlap identity (#1).

The crop-hardened embedding plateaus at keep-0.5 Rank-1 ~0.64 because a pooled global descriptor
discards the point structure a half-arch needs to register. CorrNet learns a per-point descriptor
so a partial query can be matched point-to-point against a full gallery arch (mutual-NN ->
Procrustes), the GeoTransformer-class fix. Supervision is free: sample a canonical point set,
crop it, and the surviving point indices are ground-truth correspondences. An InfoNCE loss pulls
matched descriptors together and pushes the rest apart. Trained on the 150-subject split. Saves
corrnet.pt. Needs a GPU; TP_DATA -> the poseidon3d dir.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from toothprint.identity.embedding import CorrNet

DATA = os.environ.get("TP_DATA", "data/poseidon3d/extracted/data")
WEIGHTS = Path(os.environ.get("TP_CORR_WEIGHTS", "/tmp/toothprint_embedding/corrnet.pt"))
M, DESC, N_TRAIN, EPOCHS, BATCH = 1024, 64, 150, 70, 6


def load_norm(path, n):
    m = o3d.io.read_triangle_mesh(str(path))
    if not len(m.triangles):
        return None
    p = np.asarray(m.sample_points_uniformly(n).points)
    p = p - p.mean(0)
    return (p / (np.linalg.norm(p, axis=1).max() + 1e-9)).astype(np.float32)


def rot(rng):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax); a = rng.uniform(-0.6, 0.6)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)).astype(np.float32)


def views(cloud, rng):
    """A canonical M-point set in two independently-posed views; B is a crop. bidx = GT matches."""
    idx = rng.choice(len(cloud), M, replace=len(cloud) < M)
    canon = cloud[idx]
    A = (canon @ rot(rng).T + rng.normal(0, 0.01, (M, 3))).astype(np.float32)
    keep = rng.uniform(0.4, 0.7); c = canon.mean(0); n = rng.normal(size=3); n /= np.linalg.norm(n)
    proj = (canon - c) @ n; bidx = np.where(proj >= np.quantile(proj, 1 - keep))[0]
    B = (canon[bidx] @ rot(rng).T + rng.normal(0, 0.01, (len(bidx), 3))).astype(np.float32)
    return A, B, bidx


def main():
    dev = "cuda"
    meshes = sorted(Path(DATA).glob("*/*.stl"))
    clouds = [c for c in (load_norm(m, M * 3) for m in meshes) if c is not None][:N_TRAIN]
    print(f"corrnet training on {len(clouds)} subjects, desc={DESC}", flush=True)
    net = CorrNet(DESC).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    t0 = time.time()
    for ep in range(EPOCHS):
        net.train(); rng = np.random.default_rng(ep); order = rng.permutation(len(clouds)); tot = nb = 0
        for s in range(0, len(clouds), BATCH):
            opt.zero_grad(); loss = 0.0; chunk = order[s:s + BATCH]
            for ci in chunk:
                A, B, bidx = views(clouds[ci], np.random.default_rng())
                dA = net(torch.from_numpy(A[None]).to(dev))[0]            # (D, M)
                dB = net(torch.from_numpy(B[None]).to(dev))[0]            # (D, Mb)
                logits = (dB.t() @ dA) * 20.0                            # (Mb, M) cosine logits
                loss = loss + F.cross_entropy(logits, torch.from_numpy(bidx).long().to(dev))
            loss = loss / len(chunk)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        sched.step()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"  ep {ep:3d}  loss {tot/nb:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict(), "desc": DESC, "m": M, "n_train": len(clouds)}, WEIGHTS)
    print(f"saved {WEIGHTS}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
