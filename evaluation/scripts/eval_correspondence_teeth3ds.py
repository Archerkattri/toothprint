#!/usr/bin/env python3
"""Cross-dataset generalization of the LEARNED correspondence (#1 strong test).

The earlier Teeth3DS 1.0 result used non-learned GICP (no parameters -> can't overfit). The
strong test never run: does CorrNet, trained ONLY on Poseidon3D, transfer its learned
descriptors to a DIFFERENT real intraoral dataset (Teeth3DS+) under partial overlap? Every
Teeth3DS subject is unseen by definition (different dataset). Realistic whole-tooth dropout,
same point-correspondence -> Procrustes -> all-points-residual pipeline. If it holds, the
descriptors learned arch-agnostic local geometry, not Poseidon3D idiosyncrasies. Writes
correspondence_teeth3ds.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_correspondence import WEIGHTS, crop_query, descs, rank1_auc, residual
from train_correspondence import M

from toothprint.identity.embedding import CorrNet

DATA = Path.home() / "personal-projects/toothprint-data/teeth3ds/extracted/upper"
OUT = Path(__file__).resolve().parents[1] / "results" / "correspondence_teeth3ds.json"
N_MAX, REPS = 80, 3


def load_obj(path, n):
    m = o3d.io.read_triangle_mesh(str(path))
    if not len(m.triangles):
        return None
    p = np.asarray(m.sample_points_uniformly(n).points)
    p = p - p.mean(0)
    return (p / (np.linalg.norm(p, axis=1).max() + 1e-9)).astype(np.float32)


def main():
    dev = "cuda"
    ck = torch.load(WEIGHTS, map_location=dev)
    net = CorrNet(ck["desc"]).to(dev); net.load_state_dict(ck["net"]); net.eval()
    meshes = sorted(DATA.glob("*/*.obj"))[:N_MAX]
    held = [c for c in (load_obj(m, M * 3) for m in meshes) if c is not None]
    n = len(held)
    print(f"Teeth3DS+ cross-dataset (all unseen), CorrNet trained on Poseidon3D: {n} subjects", flush=True)
    gpts = [h[np.random.default_rng(5 + i).choice(len(h), M, replace=len(h) < M)] for i, h in enumerate(held)]
    gdz = [descs(net, g, dev) for g in gpts]

    res = {"n": n, "note": "CorrNet trained on Poseidon3D, evaluated on Teeth3DS+ "
                           "(cross-dataset learned generalization, realistic whole-tooth dropout)", "results": {}}
    for keep in (0.5, 0.3):
        r1s, aucs = [], []
        for r in range(REPS):
            qpts = [crop_query(held[i], np.random.default_rng(300 + 91 * r + i), keep, "teeth") for i in range(n)]
            qdz = [descs(net, q, dev) for q in qpts]
            G = np.array([[residual(qpts[i], qdz[i], gpts[j], gdz[j]) for j in range(n)] for i in range(n)])
            r1, auc = rank1_auc(G); r1s.append(r1); aucs.append(auc)
        res["results"][f"teeth_keep{keep}"] = {"rank1": round(float(np.mean(r1s)), 3),
                                               "std": round(float(np.std(r1s)), 3), "auc": round(float(np.mean(aucs)), 3)}
        print(f"  keep {keep}: CorrNet cross-dataset Rank-1 {np.mean(r1s):.3f}±{np.std(r1s):.3f}  AUC {np.mean(aucs):.3f}"
              f"  (Poseidon3D in-domain was 0.87 / 0.57)", flush=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
