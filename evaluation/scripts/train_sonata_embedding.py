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
import json  # noqa: E402
OUT = Path(__file__).resolve().parents[1] / "results" / "sonata_identity.json"
N_PTS, EMB, BATCH = 1024, 256, 16                    # smaller batch: PTv3 is heavier than DGCNN
N_TRAIN = int(os.environ.get("TP_NTRAIN", "150"))    # train subjects; the rest are held out (unseen)
EPOCHS = int(os.environ.get("TP_EPOCHS", "80"))
FREEZE = os.environ.get("TP_FREEZE", "1") != "0"
GRID = float(os.environ.get("TP_GRID", "0.02"))
REPO_ID = os.environ.get("TP_SONATA_REPO", "facebook/sonata")


def _meshes(data):
    """All arch meshes under DATA — Poseidon3D uses .stl, Teeth3DS+ uses .obj."""
    p = Path(data)
    return sorted(list(p.glob("*/*.stl")) + list(p.glob("*/*.obj")))


@torch.no_grad()
def _embed(enc, clouds, dev):
    enc.eval()
    out = []
    for i in range(0, len(clouds), BATCH):
        x = torch.from_numpy(np.stack(clouds[i:i + BATCH])).to(dev)
        out.append(enc(x).cpu().numpy())
    return np.concatenate(out)


def _heldout_eval(enc, held, dev):
    """Rank-1 identification on UNSEEN held-out subjects at keep 1.0 / 0.5 / 0.3, same protocol as
    eval_embedding.py (repositioned + jittered + cropped genuine re-scans vs a full-arch gallery)."""
    from eval_embedding import reposition  # same query synthesis as the DGCNN eval
    n = len(held); rng = np.random.default_rng(0)
    gallery = _embed(enc, [c[rng.choice(len(c), N_PTS, replace=False)] for c in held], dev)
    res = {}
    for keep in (1.0, 0.5, 0.3):
        q = [reposition(c, np.random.default_rng(1000 + i), keep) for i, c in enumerate(held)]
        M = 1.0 - _embed(enc, q, dev) @ gallery.T                        # cosine distance
        r1 = float(np.mean([np.argmin(M[i]) == i for i in range(n)]))
        res[str(keep)] = round(r1, 3)
        print(f"  held-out keep {keep}: Sonata Rank-1 {r1:.3f}  (n={n})", flush=True)
    return res


def main():
    dev = "cuda"
    meshes = _meshes(DATA)
    print(f"loading {len(meshes)} arches ...", flush=True)
    base = [load_norm(m, N_PTS * 3) for m in meshes]
    valid = [c for c in base if c is not None]
    clouds = valid[:N_TRAIN]
    held = valid[N_TRAIN:]
    print(f"training on {len(clouds)} subjects (held-out: {len(held)})", flush=True)

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

    if held:
        ka = _heldout_eval(enc, held, dev)
        out = {"dataset": Path(DATA).name, "backbone": "sonata (PTv3, facebook/sonata SSL)",
               "n_train": len(clouds), "n_heldout": len(held), "emb_dim": EMB,
               "freeze_backbone": FREEZE, "epochs": EPOCHS, "grid_size": GRID,
               "keep_ablation_rank1": ka,
               "note": "held-out = subjects unseen in training; genuine = repositioned+jittered+cropped re-scan"}
        OUT.write_text(json.dumps(out, indent=1) + "\n")
        print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
