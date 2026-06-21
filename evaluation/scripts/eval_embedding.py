#!/usr/bin/env python3
"""Held-out evaluation of the learned 3D identity embedding.

The 52 subjects held out of training are UNSEEN people — so this measures whether the
embedding *generalises* (a real metric), not whether it memorised the gallery. Reports
Rank-1 / EER / AUC / conformal-FMR / open-set at full coverage, and the partial-overlap
(tooth-loss) curve where the classical GICP matcher collapses — run head-to-head against
GICP on the *same* held-out arches. Writes embedding_identity.json.

Run from the shared eval working dir (data/poseidon3d present); needs the trained weights.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_identity import basic_metrics, conformal_fmr, open_set
from train_embedding import DATA, N_PTS, WEIGHTS, load_norm, rand_rot

from toothprint.identity.embedding import DGCNN

OUT = Path(__file__).resolve().parents[1] / "results" / "embedding_identity.json"


def reposition(p, rng, keep, jitter=0.015):
    q = p @ rand_rot(rng).T
    if keep < 1.0:
        c = q.mean(0); nrm = rng.normal(size=3); nrm /= np.linalg.norm(nrm)
        q = q[(q - c) @ nrm >= np.quantile((q - c) @ nrm, 1 - keep)]
    idx = (rng.integers(0, len(q), N_PTS) if len(q) < N_PTS else rng.choice(len(q), N_PTS, replace=False))
    q = q[idx] + rng.normal(0, jitter, (N_PTS, 3))
    q = q - q.mean(0); q = q / (np.linalg.norm(q, axis=1).max() + 1e-9)
    return q.astype(np.float32)


@torch.no_grad()
def embed(enc, clouds, dev):
    out = []
    for i in range(0, len(clouds), 32):
        out.append(enc(torch.from_numpy(np.stack(clouds[i:i + 32])).to(dev)).cpu().numpy())
    return np.concatenate(out)


def emb_matrix(query_emb, gallery_emb):
    return 1.0 - query_emb @ gallery_emb.T              # cosine distance (unit vectors)


def main():
    dev = "cuda"
    ck = torch.load(WEIGHTS, map_location=dev)
    enc = DGCNN(ck["emb_dim"]).to(dev); enc.load_state_dict(ck["enc"]); enc.eval()
    meshes = sorted(Path(DATA).glob("*/*.stl"))
    clouds = [c for c in (load_norm(m, N_PTS * 3) for m in meshes) if c is not None]
    held = clouds[ck["n_train"]:]                       # UNSEEN subjects
    n = len(held); rng = np.random.default_rng(0)
    print(f"held-out (unseen) subjects: {n}", flush=True)

    gallery = embed(enc, [c[rng.choice(len(c), N_PTS, replace=False)] for c in held], dev)
    # GICP whole-arch baseline keep-ablation (committed id3d.json, full set) for the head-to-head
    gabl = json.loads((OUT.parent / "id3d.json").read_text())["ablations"]
    gicp_keep = {1.0: gabl["keep_1.0"]["rank1"], 0.5: gabl["keep_0.5"]["rank1"], 0.3: gabl["keep_0.3"]["rank1"]}
    res = {"n_heldout": n, "emb_dim": ck["emb_dim"], "method": "DGCNN + sub-centre ArcFace",
           "note": "held-out = subjects unseen in training; gicp_rank1 is the committed whole-arch baseline",
           "keep_ablation": {}}

    for keep in [1.0, 0.5, 0.3]:
        q = [reposition(c, np.random.default_rng(1000 + i), keep) for i, c in enumerate(held)]
        M = emb_matrix(embed(enc, q, dev), gallery)
        emb_r1 = float(np.mean([np.argmin(M[i]) == i for i in range(n)]))
        res["keep_ablation"][str(keep)] = {"embedding_rank1": emb_r1, "gicp_rank1": gicp_keep[keep]}
        print(f"  keep {keep}: embedding Rank-1 {emb_r1:.3f} | GICP whole-arch {gicp_keep[keep]:.3f}", flush=True)
        if keep == 1.0:
            bm = basic_metrics(M)
            res["main"] = {k: bm[k] for k in ("n", "rank1", "rank5", "auc", "eer", "dprime")}
            conf = conformal_fmr(M, [0.01, 0.05, 0.1])
            res["conformal_fmr"] = {str(a): conf[a] for a in conf}
            res["open_set"] = {"fnir_at_fpir_1pct": open_set(M, held_out=min(20, n // 3))["fnir_at_fpir_1pct"]}
            print(f"  full-coverage: Rank-1 {bm['rank1']:.3f} EER {bm['eer']:.3f} AUC {bm['auc']:.3f}", flush=True)

    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
