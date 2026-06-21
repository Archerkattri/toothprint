#!/usr/bin/env python3
"""Cross-dataset identity on Teeth3DS+ — does the method generalise off Poseidon3D?

Teeth3DS+ (MICCAI 3DTeethSeg, open on OSF) is a *different* real intraoral-scan dataset.
Running the same PCA-init + GICP identity on its upper-jaw scans tests whether the 0.995
Poseidon3D result is dataset-specific or a real property of the method. Synthetic genuine
re-scans (single scan per subject), same as the Poseidon3D protocol. Writes
teeth3ds_identity.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_id3d import KEEP, NOISE, NPTS, ROT_AMP, VX, build_matrix, metrics

DATA = Path.home() / "personal-projects/toothprint-data/teeth3ds/extracted/upper"
OUT = Path(__file__).resolve().parents[1] / "results" / "teeth3ds_identity.json"
N_MAX = 120


def main():
    meshes = sorted(DATA.glob("*/*.obj"))[:N_MAX]
    workers = max(1, min(16, (os.cpu_count() or 2) - 2))
    print(f"Teeth3DS+ upper-jaw scans: {len(meshes)} subjects, {workers} workers ...", flush=True)
    M, labels = build_matrix(meshes, NOISE, ROT_AMP, KEEP, VX, NPTS, workers)
    m = metrics(M, labels)
    out = {"dataset": "Teeth3DS+ (real intraoral, cross-dataset generalisation)",
           "n": m["n"], "rank1": m["rank1"], "rank5": m["rank5"], "eer": m["eer"], "auc": m["auc"],
           "genuine_mean": m["genuine_mean"], "impostor_min": m["impostor_min"]}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  N={m['n']}  Rank-1 {m['rank1']:.3f}  Rank-5 {m['rank5']:.3f}  EER {m['eer']:.3f}  AUC {m['auc']:.3f}", flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
