#!/usr/bin/env python3
"""Train the learned 3D identity embedding (DGCNN + sub-centre ArcFace) on Poseidon3D.

Trained on a **subject split** (first 150 arches); the last 52 are held out entirely so
eval_embedding.py measures generalisation to *unseen people*, not memorisation. Genuine
variation is synthesised per step — full-SO(3) rotation, jitter, partial-arch crop — so the
encoder must learn a pose/coverage-invariant metric. Saves the encoder weights (gitignored,
regenerable); the committed artifact is the eval JSON.

Run from the shared eval working dir (data/poseidon3d present); needs a GPU.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader, Dataset

from toothprint.identity.embedding import DGCNN, SubCenterArcFace

DATA = "data/poseidon3d/extracted/data"
WEIGHTS = Path("/tmp/toothprint_embedding/encoder.pt")
N_PTS, EMB, N_TRAIN, EPOCHS, BATCH = 1024, 256, 150, 80, 32


def load_norm(path, n):
    m = o3d.io.read_triangle_mesh(str(path))
    if not len(m.triangles):
        return None
    p = np.asarray(m.sample_points_uniformly(n).points)
    p = p - p.mean(0); p = p / (np.linalg.norm(p, axis=1).max() + 1e-9)
    return p.astype(np.float32)


def rand_rot(rng):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax); a = rng.uniform(-0.6, 0.6)  # realistic reposition
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)).astype(np.float32)


def augment(p, rng, jitter=0.015, keep_lo=0.6):
    q = p @ rand_rot(rng).T
    kf = rng.uniform(keep_lo, 1.0)
    if kf < 1.0:
        c = q.mean(0); n = rng.normal(size=3); n /= np.linalg.norm(n)
        q = q[(q - c) @ n >= np.quantile((q - c) @ n, 1 - kf)]
    idx = (rng.integers(0, len(q), N_PTS) if len(q) < N_PTS else rng.choice(len(q), N_PTS, replace=False))
    q = q[idx] + rng.normal(0, jitter, (N_PTS, 3)).astype(np.float32)
    q = q - q.mean(0); q = q / (np.linalg.norm(q, axis=1).max() + 1e-9)
    return q.astype(np.float32)


class ArchDS(Dataset):
    def __init__(self, clouds, reps=8):
        self.c = clouds; self.reps = reps

    def __len__(self):
        return len(self.c) * self.reps                         # many augmented views / subject / epoch

    def __getitem__(self, i):
        i %= len(self.c)
        rng = np.random.default_rng()                          # fresh entropy each step
        return torch.from_numpy(augment(self.c[i], rng)), i


def main():
    dev = "cuda"
    meshes = sorted(Path(DATA).glob("*/*.stl"))
    print(f"loading {len(meshes)} arches ...", flush=True)
    base = [load_norm(m, N_PTS * 3) for m in meshes]
    clouds = [c for c in base if c is not None][:N_TRAIN]
    print(f"training on {len(clouds)} subjects (held-out: {len([c for c in base if c is not None]) - len(clouds)})", flush=True)

    enc = DGCNN(EMB).to(dev); head = SubCenterArcFace(EMB, len(clouds)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    dl = DataLoader(ArchDS(clouds), batch_size=BATCH, shuffle=True, num_workers=6, drop_last=True)

    t0 = time.time()
    for ep in range(EPOCHS):
        enc.train(); head.train(); tot = cor = nseen = 0
        head.m = min(0.4, 0.4 * (ep + 1) / 20)                 # margin warmup — avoid ArcFace cold-start
        for pts, lab in dl:
            pts, lab = pts.to(dev), lab.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                emb = enc(pts); logits = head(emb, lab)
                loss = torch.nn.functional.cross_entropy(logits, lab)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(lab); nseen += len(lab)
            cor += (head.cosine(emb).argmax(1) == lab).sum().item()    # un-margined retrieval acc
        sched.step()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"  ep {ep:3d}  loss {tot/nseen:.3f}  train-acc {cor/nseen:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"enc": enc.state_dict(), "emb_dim": EMB, "n_pts": N_PTS, "n_train": len(clouds)}, WEIGHTS)
    print(f"saved {WEIGHTS}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
