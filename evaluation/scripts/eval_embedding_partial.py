#!/usr/bin/env python3
"""Head-to-head (#1): does crop-hardened training improve partial-overlap identity?

The partial-overlap probe showed the bottleneck is the *descriptor*, not the scoring — and the
baseline embedding was trained with only mild crops (keep>=0.6) yet is tested at 50-70% tooth
loss. This evaluates the crop-hardened encoder (keep>=0.35) against the baseline on the SAME
unseen held-out subjects, the SAME gallery clouds, and the SAME repositioned queries — so the
encoder is the only variable. Reports Rank-1 at keep {1.0, 0.5, 0.3} for each. Writes
embedding_partial.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_embedding import emb_matrix, embed, reposition
from train_embedding import DATA, N_PTS, load_norm

from toothprint.identity.embedding import DGCNN

OUT = Path(__file__).resolve().parents[1] / "results" / "embedding_partial.json"
BASE = Path("/tmp/toothprint_embedding/encoder.pt")
HARD = Path("/tmp/toothprint_embedding/encoder_partial.pt")


def load_enc(path, dev):
    ck = torch.load(path, map_location=dev)
    enc = DGCNN(ck["emb_dim"]).to(dev); enc.load_state_dict(ck["enc"]); enc.eval()
    return enc, ck["n_train"]


def rank1(M):
    return float(np.mean([np.argmin(M[i]) == i for i in range(len(M))]))


def main():
    dev = "cuda"
    encB, ntr = load_enc(BASE, dev)
    encH, _ = load_enc(HARD, dev)
    meshes = sorted(Path(DATA).glob("*/*.stl"))
    clouds = [c for c in (load_norm(m, N_PTS * 3) for m in meshes) if c is not None]
    held = clouds[ntr:]; n = len(held)
    print(f"held-out unseen subjects: {n}", flush=True)

    gal_clouds = [c[np.random.default_rng(500 + i).choice(len(c), N_PTS, replace=False)]    # identical for both
                  for i, c in enumerate(held)]
    galB, galH = embed(encB, gal_clouds, dev), embed(encH, gal_clouds, dev)
    REPS = 4                                                   # avg over query realizations: effective N=200, kills the N=50 noise
    res = {"n_heldout": n, "query_reps": REPS, "keep_ablation": {}}
    for keep in [1.0, 0.5, 0.3]:
        rB, rH, rE = [], [], []
        for r in range(REPS):
            q = [reposition(c, np.random.default_rng(1000 + 137 * r + i), keep) for i, c in enumerate(held)]
            MB = emb_matrix(embed(encB, q, dev), galB)
            MH = emb_matrix(embed(encH, q, dev), galH)
            rB.append(rank1(MB)); rH.append(rank1(MH)); rE.append(rank1(MB + MH))   # ensemble = score sum
        r1B, r1H, r1E = float(np.mean(rB)), float(np.mean(rH)), float(np.mean(rE))
        res["keep_ablation"][str(keep)] = {"baseline_rank1": round(r1B, 3), "baseline_std": round(float(np.std(rB)), 3),
                                           "crop_hardened_rank1": round(r1H, 3),
                                           "ensemble_rank1": round(r1E, 3), "ensemble_std": round(float(np.std(rE)), 3),
                                           "ensemble_gain_over_baseline": round(r1E - r1B, 3)}
        print(f"  keep {keep}: baseline {r1B:.3f} | crop-hardened {r1H:.3f} | ensemble {r1E:.3f}"
              f"   (ens vs base {r1E - r1B:+.3f}, ens std {np.std(rE):.3f})", flush=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
