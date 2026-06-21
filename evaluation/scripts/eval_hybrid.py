#!/usr/bin/env python3
"""Hybrid identity (#3): retrieve-by-embedding -> refine-by-GICP.

The learned embedding is partial-overlap robust but loses to GICP at full coverage; GICP is
precise but collapses under tooth loss. The hybrid takes the best of both: embed the query,
shortlist the top-K gallery arches by embedding distance (robust retrieval), then re-rank
that shortlist by GICP surface distance (precise refinement). Evaluated on the SAME 50
held-out (unseen) subjects, at full + partial coverage, vs embedding-alone. Writes
hybrid_identity.json. Needs the trained encoder + a GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_embedding import embed, emb_matrix, reposition
from train_embedding import DATA, N_PTS, WEIGHTS, load_norm, rand_rot

from toothprint.identity import align_rigid
from toothprint.identity.embedding import DGCNN

OUT = Path(__file__).resolve().parents[1] / "results" / "hybrid_identity.json"
K = 5


def sample_metric(path, n):
    m = o3d.io.read_triangle_mesh(str(path))
    return np.asarray(m.sample_points_uniformly(n).points).astype(np.float32) if len(m.triangles) else None


def reposition_metric(p, rng, keep, noise=0.06):
    q = p @ rand_rot(rng).T + rng.uniform(-5, 5, 3)
    if keep < 1.0:
        c = q.mean(0); nr = rng.normal(size=3); nr /= np.linalg.norm(nr)
        q = q[(q - c) @ nr >= np.quantile((q - c) @ nr, 1 - keep)]
    return (q + rng.normal(0, noise, q.shape)).astype(np.float32)


def main():
    dev = "cuda"
    ck = torch.load(WEIGHTS, map_location=dev)
    enc = DGCNN(ck["emb_dim"]).to(dev); enc.load_state_dict(ck["enc"]); enc.eval()
    meshes = sorted(Path(DATA).glob("*/*.stl"))
    pairs = []
    for m in meshes:
        nc, mc = load_norm(m, N_PTS * 3), sample_metric(m, N_PTS * 3)
        if nc is not None and mc is not None:
            pairs.append((nc, mc))
    held = pairs[ck["n_train"]:]
    norm_h = [p[0] for p in held]; metric_h = [p[1] for p in held]
    n = len(held); rng = np.random.default_rng(0)
    print(f"held-out (unseen) subjects: {n}", flush=True)

    gallery = embed(enc, [c[rng.choice(len(c), N_PTS, replace=False)] for c in norm_h], dev)
    gabl = json.loads((OUT.parent / "id3d.json").read_text())["ablations"]
    gicp_keep = {1.0: gabl["keep_1.0"]["rank1"], 0.5: gabl["keep_0.5"]["rank1"], 0.3: gabl["keep_0.3"]["rank1"]}
    res = {"n_heldout": n, "topK": K, "method": "embedding retrieve -> GICP refine", "keep_ablation": {}}

    for keep in [1.0, 0.5, 0.3]:
        emb_ok = hyb_ok = 0
        for i in range(n):
            nq = reposition(norm_h[i], np.random.default_rng(2000 + i), keep)
            mq = reposition_metric(metric_h[i], np.random.default_rng(3000 + i), keep)
            ed = emb_matrix(embed(enc, [nq], dev), gallery)[0]
            topk = np.argsort(ed)[:K]
            emb_ok += int(np.argmin(ed) == i)
            gd = [align_rigid(mq, metric_h[j], 0.5)[1] for j in topk]
            hyb_ok += int(topk[int(np.argmin(gd))] == i)
        emb_r1, hyb_r1 = emb_ok / n, hyb_ok / n
        res["keep_ablation"][str(keep)] = {"embedding_rank1": emb_r1, "hybrid_rank1": hyb_r1,
                                            "gicp_wholearch_rank1": gicp_keep[keep]}
        print(f"  keep {keep}: embedding {emb_r1:.3f} -> HYBRID {hyb_r1:.3f}  (GICP whole-arch {gicp_keep[keep]:.3f})", flush=True)

    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
