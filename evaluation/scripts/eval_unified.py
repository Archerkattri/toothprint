#!/usr/bin/env python3
"""Unified certified accept/abstain identity decision — the matcher assembled from the parts.

One pipeline, one verdict: retrieve candidates by the learned EMBEDDING (partial-robust recall),
verify the shortlist by CorrNet point CORRESPONDENCE (the residual that survives tooth loss),
then ACCEPT the top candidate only if its residual clears a conformal threshold (empirical
FMR <= alpha on held-out impostors) — otherwise ABSTAIN. Embedding for recall, correspondence for
precision, conformal for the guarantee.

Measured open-set (enrolled genuine + non-enrolled impostor) at full AND partial coverage. The
hypothesis: because CorrNet's residual separates genuine from impostor well even under tooth loss
(AUC ~0.98 at keep-0.5), the unified decision should reject impostors at partial overlap far
better than the embedding/GICP open-set that collapsed there (#3). Writes unified_identity.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths
from eval_correspondence import WEIGHTS as CORR_W, crop_query, descs, residual
from eval_openset_hybrid import fnir_at_fpir
from train_correspondence import M, load_norm

from toothprint.identity.embedding import CorrNet, DGCNN

EMB_W = Path("/tmp/toothprint_embedding/encoder_partial.pt")
OUT = Path(__file__).resolve().parents[1] / "results" / "unified_identity.json"
TOPK, HELD_OUT, TRIALS = 5, 15, 40


@torch.no_grad()
def emb_vecs(enc, clouds, dev):
    out = []
    for i in range(0, len(clouds), 32):
        b = np.stack([c[np.random.default_rng(7).choice(len(c), 1024, replace=len(c) < 1024)] for c in clouds[i:i + 32]])
        out.append(enc(torch.from_numpy(b).to(dev)).cpu().numpy())
    return np.concatenate(out)


def main():
    dev = "cuda"
    eck = torch.load(EMB_W, map_location=dev); enc = DGCNN(eck["emb_dim"]).to(dev); enc.load_state_dict(eck["enc"]); enc.eval()
    cck = torch.load(CORR_W, map_location=dev); cnet = CorrNet(cck["desc"]).to(dev); cnet.load_state_dict(cck["net"]); cnet.eval()
    meshes = sorted(Path(str(paths.POSEIDON3D)).glob("*/*.stl"))
    clouds = [c for c in (load_norm(m, M * 3) for m in meshes) if c is not None]
    held = clouds[cck["n_train"]:]; n = len(held)
    print(f"unified decision: {n} held-out unseen subjects", flush=True)
    gpts = [h[np.random.default_rng(5 + i).choice(len(h), M, replace=len(h) < M)] for i, h in enumerate(held)]
    gdesc = [descs(cnet, g, dev) for g in gpts]
    gemb = emb_vecs(enc, gpts, dev)

    res = {"n_heldout": n, "topk": TOPK,
           "pipeline": "embedding-retrieve(top-5) -> CorrNet-verify -> conformal accept/abstain",
           "coverage": {}}
    for keep in (1.0, 0.5):
        qpts = [held[i][np.random.default_rng(700 + i).choice(len(held[i]), M, replace=len(held[i]) < M)] if keep >= 1.0
                else crop_query(held[i], np.random.default_rng(700 + i), keep, "teeth") for i in range(n)]
        qemb = emb_vecs(enc, qpts, dev); qdesc = [descs(cnet, q, dev) for q in qpts]
        fn = []
        for tr in range(TRIALS):
            perm = np.random.default_rng(tr).permutation(n); non, enr = list(perm[:HELD_OUT]), list(perm[HELD_OUT:])
            enr_arr = np.array(enr)
            gs, gc, ns = [], [], []
            for i in enr + non:
                sims = gemb[enr_arr] @ qemb[i]                                # embedding retrieval among enrolled
                sh = enr_arr[np.argsort(-sims)[:TOPK]]
                rs = [residual(qpts[i], qdesc[i], gpts[j], gdesc[j]) for j in sh]
                k = int(np.argmin(rs)); score, jhat = rs[k], int(sh[k])
                if i in non:
                    ns.append(score)
                else:
                    gs.append(score); gc.append(jhat == i)
            fn.append(fnir_at_fpir(np.array(gs), np.array(gc), np.array(ns)))
        res["coverage"][str(keep)] = {"unified_fnir_at_fmr1pct": round(float(np.mean(fn)), 3), "std": round(float(np.std(fn)), 3)}
        print(f"  coverage {keep}: unified FNIR@FMR=1% {np.mean(fn):.3f} ± {np.std(fn):.3f}", flush=True)
    res["reference_openset_keep0.5"] = {"embedding_only": 0.872, "gicp_only": 0.779, "hybrid_gicp_decide": 0.789,
                                        "source": "openset_hybrid.json — the methods that collapsed at partial overlap"}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
