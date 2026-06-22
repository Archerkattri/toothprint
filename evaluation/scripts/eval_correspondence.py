#!/usr/bin/env python3
"""Eval CorrNet partial->whole correspondence identity (#1) vs the crop-hardened ceiling.

Each 50%-cropped query is matched point-to-point against every full gallery arch by CorrNet
descriptors (mutual nearest neighbours), a weighted Procrustes fit aligns it, and the residual
scores the match — a genuine half-arch finds dense, consistent correspondences (low residual),
an impostor cannot. Rank-1 at keep-0.5 head-to-head with the crop-hardened embedding (~0.64) and
rigid GICP (~0.23) on the SAME held-out unseen subjects. Writes correspondence_identity.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_correspondence import DATA, M, load_norm, rot

from toothprint.identity.embedding import CorrNet

WEIGHTS = Path("/tmp/toothprint_embedding/corrnet.pt")
OUT = Path(__file__).resolve().parents[1] / "results" / "correspondence_identity.json"
KEEP, REPS = 0.5, 3


def crop_query(cloud, rng, keep=KEEP, mode="planar"):
    """Partial query. mode='planar' = clean half-space cut (easy); mode='teeth' = REALISTIC
    discrete whole-tooth dropout (random, often non-contiguous units — what actually happens)."""
    idx = rng.choice(len(cloud), M, replace=len(cloud) < M)
    p = cloud[idx] @ rot(rng).T
    c = p.mean(0)
    if mode == "planar":
        n = rng.normal(size=3); n /= np.linalg.norm(n)
        proj = (p - c) @ n; p = p[proj >= np.quantile(proj, 1 - keep)]
    else:                                                                    # discrete whole-tooth dropout
        u = np.linalg.svd(p - c, full_matrices=False)[2][0]                  # arch axis
        proj = (p - c) @ u
        band = np.clip(np.digitize(proj, np.linspace(proj.min(), proj.max(), 15)[1:-1]), 0, 13)
        teeth = np.unique(band); nk = max(2, int(round(keep * len(teeth))))
        kept = rng.choice(teeth, nk, replace=False)                          # random teeth, not a clean half
        p = p[np.isin(band, kept)]
    return (p + rng.normal(0, 0.01, p.shape)).astype(np.float32)


@torch.no_grad()
def descs(net, cloud, dev):
    return net(torch.from_numpy(cloud[None]).to(dev))[0].cpu().numpy().T          # (P, D)


def procrustes(q, g, w):
    wq = (w[:, None] * q).sum(0) / w.sum(); wg = (w[:, None] * g).sum(0) / w.sum()
    H = ((q - wq) * w[:, None]).T @ (g - wg); U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, wg - R @ wq


def residual(qpts, qd, gpts, gd):
    """Honest score: fit Procrustes on MUTUAL matches, then measure ALL query points against
    their descriptor match after that alignment (so the fit can't flatter itself)."""
    sim = qd @ gd.T                                                             # (Pq, Pg) cosine
    gj = sim.argmax(1); conf = sim.max(1)
    mut = sim.argmax(0)[gj] == np.arange(len(qd))                              # mutual NN
    if mut.sum() < 10:
        return 1e6
    R, t = procrustes(qpts[mut], gpts[gj[mut]], conf[mut])
    return float(np.linalg.norm((qpts @ R.T + t) - gpts[gj], axis=1).mean())   # ALL query points, not just fitted ones


def rank1_auc(G):
    n = len(G); r1 = float(np.mean([np.argmin(G[i]) == i for i in range(n)]))
    gen = np.array([G[i, i] for i in range(n)]); imp = np.array([G[i, j] for i in range(n) for j in range(n) if i != j])
    return r1, float(np.mean(gen[:, None] < imp[None, :]))


def main():
    dev = "cuda"
    ck = torch.load(WEIGHTS, map_location=dev)
    net = CorrNet(ck["desc"]).to(dev); net.load_state_dict(ck["net"]); net.eval()
    meshes = sorted(Path(DATA).glob("*/*.stl"))
    clouds = [c for c in (load_norm(m, M * 3) for m in meshes) if c is not None]
    held = clouds[ck["n_train"]:]; n = len(held)
    print(f"corrnet eval: {n} held-out unseen subjects ...", flush=True)
    gpts = [h[np.random.default_rng(5 + i).choice(len(h), M, replace=len(h) < M)] for i, h in enumerate(held)]
    gdz = [descs(net, g, dev) for g in gpts]

    res = {"n_heldout": n, "reps": REPS, "score": "all-query-points residual after mutual-match Procrustes",
           "crop_modes": {"planar": "clean half-space cut (easy)", "teeth": "discrete random whole-tooth dropout (realistic)"},
           "results": {}}
    for mode in ("planar", "teeth"):
        for keep in (0.5, 0.3):
            r1s, aucs = [], []
            for r in range(REPS):
                qpts = [crop_query(held[i], np.random.default_rng(300 + 91 * r + i), keep, mode) for i in range(n)]
                qdz = [descs(net, q, dev) for q in qpts]
                G = np.array([[residual(qpts[i], qdz[i], gpts[j], gdz[j]) for j in range(n)] for i in range(n)])
                r1, auc = rank1_auc(G); r1s.append(r1); aucs.append(auc)
            res["results"][f"{mode}_keep{keep}"] = {"corrnet_rank1": round(float(np.mean(r1s)), 3),
                                                    "std": round(float(np.std(r1s)), 3), "auc": round(float(np.mean(aucs)), 3)}
            print(f"  {mode:6s} keep {keep}: CorrNet Rank-1 {np.mean(r1s):.3f}±{np.std(r1s):.3f}  AUC {np.mean(aucs):.3f}", flush=True)
    def _ref(name, getter, default):                                          # read baselines from committed artifacts, not pasted constants
        try:
            return round(float(getter(json.loads((OUT.parent / name).read_text()))), 3)
        except Exception:
            return default
    res["baselines_planar_reference"] = {
        "source": "embedding_partial.json (crop-hardened embedding), id3d.json (rigid GICP) — read at runtime",
        "keep0.5": {"crop_hardened": _ref("embedding_partial.json", lambda d: d["keep_ablation"]["0.5"]["crop_hardened_rank1"], 0.635),
                    "rigid_gicp": _ref("id3d.json", lambda d: d["ablations"]["keep_0.5"]["rank1"], 0.23)},
        "keep0.3": {"crop_hardened": _ref("embedding_partial.json", lambda d: d["keep_ablation"]["0.3"]["crop_hardened_rank1"], 0.26),
                    "rigid_gicp": _ref("id3d.json", lambda d: d["ablations"]["keep_0.3"]["rank1"], 0.10)}}

    torch.manual_seed(0)                                                       # untrained control: isolates pipeline vs learning
    rnet = CorrNet(ck["desc"]).to(dev).eval()
    rgdz = [descs(rnet, g, dev) for g in gpts]
    cr = []
    for r in range(REPS):
        qpts = [crop_query(held[i], np.random.default_rng(300 + 91 * r + i), 0.5) for i in range(n)]
        qdz = [descs(rnet, q, dev) for q in qpts]
        cr.append(rank1_auc(np.array([[residual(qpts[i], qdz[i], gpts[j], rgdz[j]) for j in range(n)] for i in range(n)]))[0])
    res["untrained_control_keep0.5_rank1"] = round(float(np.mean(cr)), 3)
    print(f"  CONTROL untrained CorrNet keep-0.5 Rank-1 {np.mean(cr):.3f}  (the correspondence ARCHITECTURE alone; "
          f"learning adds the rest)", flush=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
