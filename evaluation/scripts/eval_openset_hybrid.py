#!/usr/bin/env python3
"""#3: route the open-set reject decision through GICP, not the embedding.

The learned embedding retrieves well but REJECTS non-enrolled queries poorly (open-set
FNIR@FPIR=1% ~0.69) — a cosine score doesn't separate impostors. GICP surface distance does
(FNIR ~0.03), but its RETRIEVAL collapses under partial overlap. The hybrid uses each tool for
its job: the embedding shortlists the top-k gallery arches (partial-robust retrieval), then
GICP scores the shortlist (sharp, conformally-bounded accept/reject). Compared head-to-head
with embedding-only and GICP-only open-set on partial (keep-0.5) queries of held-out UNSEEN
subjects, over many enrolled/non-enrolled splits. Writes openset_hybrid.json.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths
from eval_id3d import NPTS, VX, sample
from train_embedding import N_PTS

from toothprint.identity import align_rigid
from toothprint.identity.embedding import DGCNN

DATA = str(paths.POSEIDON3D)
HARD = Path("/tmp/toothprint_embedding/encoder_partial.pt")
OUT = Path(__file__).resolve().parents[1] / "results" / "openset_hybrid.json"
N, KEEP, TOPK, HELD_OUT, TRIALS = 50, 0.5, 5, 15, 60

_GAL = None


def _init(gal):
    global _GAL
    _GAL = gal


def _grow(task):
    i, q = task
    return i, np.array([align_rigid(q, g, VX)[1] for g in _GAL])      # GICP distance row


def norm_unit(p):
    p = p - p.mean(0)
    return (p / (np.linalg.norm(p, axis=1).max() + 1e-9)).astype(np.float32)


def crop(p, rng, keep):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax); a = rng.uniform(-0.3, 0.3)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    q = p @ (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)).T + rng.uniform(-5, 5, 3)
    c = q.mean(0); n = rng.normal(size=3); n /= np.linalg.norm(n)
    return q[(q - c) @ n >= np.quantile((q - c) @ n, 1 - keep)]                             # noise added in main


@torch.no_grad()
def embed(enc, clouds, dev):
    out = []
    for i in range(0, len(clouds), 32):
        batch = np.stack([c[np.random.default_rng(7).choice(len(c), N_PTS, len(c) < N_PTS)] for c in clouds[i:i + 32]])
        out.append(enc(torch.from_numpy(batch).to(dev)).cpu().numpy())
    return np.concatenate(out)


def fnir_at_fpir(gscore, gcorrect, nscore, taus=400):
    lo = min(gscore.min(), nscore.min()); hi = max(np.percentile(gscore, 97), np.percentile(nscore, 60))
    grid = np.linspace(lo, hi, taus)
    dirc = np.array([np.mean((gscore < t) & gcorrect) for t in grid])
    fpic = np.array([np.mean(nscore < t) for t in grid])
    order = np.argsort(fpic)
    return float(1 - np.interp(0.01, fpic[order], dirc[order]))


def openset(E, G, rng):
    """Average open-set FNIR@FPIR=1% over enrolled/non-enrolled splits for each method."""
    n = len(E); fn = {"embedding_only": [], "gicp_only": [], "hybrid": []}
    for _ in range(TRIALS):
        perm = rng.permutation(n); non, enr = list(perm[:HELD_OUT]), list(perm[HELD_OUT:])
        for meth in fn:
            gs, gc, ns = [], [], []
            for i in enr:
                if meth == "embedding_only":
                    s = E[i, enr]; jhat = enr[int(np.argmin(s))]; sc = float(s.min())
                elif meth == "gicp_only":
                    s = G[i, enr]; jhat = enr[int(np.argmin(s))]; sc = float(s.min())
                else:                                                         # hybrid: embed-retrieve -> gicp-decide
                    sh = [enr[k] for k in np.argsort(E[i, enr])[:TOPK]]
                    gj = G[i, sh]; jhat = sh[int(np.argmin(gj))]; sc = float(gj.min())
                gs.append(sc); gc.append(jhat == i)
            for i in non:
                if meth == "embedding_only":
                    ns.append(float(E[i, enr].min()))
                elif meth == "gicp_only":
                    ns.append(float(G[i, enr].min()))
                else:
                    sh = [enr[k] for k in np.argsort(E[i, enr])[:TOPK]]; ns.append(float(G[i, sh].min()))
            fn[meth].append(fnir_at_fpir(np.array(gs), np.array(gc), np.array(ns)))
    return {m: round(float(np.mean(v)), 3) for m, v in fn.items()}


def main():
    dev = "cuda"
    ck = torch.load(HARD, map_location=dev)
    enc = DGCNN(ck["emb_dim"]).to(dev); enc.load_state_dict(ck["enc"]); enc.eval()
    meshes = sorted(Path(DATA).glob("*/*.stl"))
    raw = [r for r in (sample(m, NPTS) for m in meshes) if r is not None][ck["n_train"]:][:N]    # held-out unseen
    n = len(raw); rng = np.random.default_rng(0)
    print(f"open-set hybrid: {n} held-out unseen subjects, keep={KEEP} ...", flush=True)

    queries = [crop(raw[i], np.random.default_rng(900 + i), KEEP) for i in range(n)]
    queries = [q + np.random.default_rng(900 + i).normal(0, 0.06, q.shape) for i, q in enumerate(queries)]
    galE = embed(enc, [norm_unit(g) for g in raw], dev)
    qE = embed(enc, [norm_unit(q) for q in queries], dev)
    E = 1.0 - qE @ galE.T                                                 # embedding cosine distance

    workers = max(1, min(16, (mp.cpu_count() or 2) - 2))
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(raw,)) as pool:
        rows = pool.map(_grow, [(i, queries[i]) for i in range(n)])
    G = np.zeros((n, n))
    for i, r in rows:
        G[i] = r

    res = {"n_heldout": n, "keep": KEEP, "topk": TOPK, "prior_full_coverage": {"embedding_fnir": 0.69, "gicp_fnir": 0.03},
           "partial_open_set_fnir_at_fpir1pct": openset(E, G, rng)}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    for m, v in res["partial_open_set_fnir_at_fpir1pct"].items():
        print(f"  {m:16s} FNIR@FPIR=1% {v:.3f}", flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
