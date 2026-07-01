#!/usr/bin/env python3
"""Train the learned 3D identity embedding on a **Sonata-pretrained PTv3** backbone (#1 upgrade).

Mirrors train_embedding.py exactly — same Poseidon3D **subject split** (first 150 arches, last
52 held out for unseen-people generalisation), same full-SO(3) + jitter + partial-arch crop
augmentation, same sub-centre ArcFace head + margin warmup — but swaps the from-scratch DGCNN for
Point Transformer V3 pretrained by Sonata self-supervised learning (Wu et al., CVPR 2025, arXiv
2503.16429; weights ``facebook/sonata`` on HuggingFace, code in Pointcept). This is the first
application of a point-cloud foundation model to dental identity.

By default the Sonata encoder is **frozen** and only the projection head + ArcFace centres train
(the right recipe for 150 subjects); set ``TP_FREEZE=0`` to fine-tune the whole encoder end-to-end.

Saves the encoder weights (gitignored, regenerable); the committed artifact is the eval JSON.
Needs a GPU **and** a working Pointcept install — see evaluation/scripts/RUN.md for the exact,
tested setup and the expected memory/throughput. If Pointcept is missing this script fails fast
with an install hint; it never fabricates results.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader, Dataset

from toothprint.identity.embedding import SubCenterArcFace, build_embedding_backbone

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

# Reuse the DGCNN recipe verbatim so the comparison is backbone-only.
from train_embedding import ArchDS, augment, load_norm, rand_rot  # noqa: E402,F401

DATA = os.environ.get("TP_DATA", str(paths.POSEIDON3D))
WEIGHTS = Path(os.environ.get("TP_SONATA_WEIGHTS", "/tmp/toothprint_embedding/sonata_encoder.pt"))
N_PTS, EMB, N_TRAIN, BATCH = 1024, 256, 150, 16      # smaller batch: PTv3 is heavier than DGCNN
EPOCHS = int(os.environ.get("TP_EPOCHS", "80"))
FREEZE = os.environ.get("TP_FREEZE", "1") != "0"
GRID = float(os.environ.get("TP_GRID", "0.02"))
REPO_ID = os.environ.get("TP_SONATA_REPO", "facebook/sonata")


def main():
    dev = "cuda"
    meshes = sorted(Path(DATA).glob("*/*.stl"))
    print(f"loading {len(meshes)} arches ...", flush=True)
    base = [load_norm(m, N_PTS * 3) for m in meshes]
    clouds = [c for c in base if c is not None][:N_TRAIN]
    print(f"training on {len(clouds)} subjects (held-out: "
          f"{len([c for c in base if c is not None]) - len(clouds)})", flush=True)

    enc = build_embedding_backbone("sonata", emb_dim=EMB, grid_size=GRID, repo_id=REPO_ID,
                                   freeze_backbone=FREEZE).to(dev)
    enc.load()                                              # fetch/build PTv3 (clear error if no Pointcept)
    head = SubCenterArcFace(EMB, len(clouds)).to(dev)
    params = [p for p in enc.parameters() if p.requires_grad] + list(head.parameters())
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f}M  (freeze_backbone={FREEZE})",
          flush=True)
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    dl = DataLoader(ArchDS(clouds), batch_size=BATCH, shuffle=True, num_workers=6, drop_last=True)

    t0 = time.time()
    for ep in range(EPOCHS):
        head.train()
        enc.head.train()
        tot = cor = nseen = 0
        head.m = min(0.4, 0.4 * (ep + 1) / 20)             # margin warmup — avoid ArcFace cold-start
        for pts, lab in dl:
            pts, lab = pts.to(dev), lab.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                emb = enc(pts)
                logits = head(emb, lab)
                loss = torch.nn.functional.cross_entropy(logits, lab)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(lab); nseen += len(lab)
            cor += (head.cosine(emb).argmax(1) == lab).sum().item()
        sched.step()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"  ep {ep:3d}  loss {tot/nseen:.3f}  train-acc {cor/nseen:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"enc": enc.state_dict(), "backbone": "sonata", "emb_dim": EMB, "n_pts": N_PTS,
                "n_train": len(clouds), "grid_size": GRID, "freeze_backbone": FREEZE}, WEIGHTS)
    print(f"saved {WEIGHTS}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
