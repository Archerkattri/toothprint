#!/usr/bin/env python3
"""FULL-COVERAGE identity with **BUFFER-X** as the registration/scoring backend — the complete
Rank-1 / Rank-5 / EER / AUC column to sit beside the PCA-init+GICP smoke
(``teeth3ds_identity_smoke_n40.json``) on the SAME N=40 real Teeth3DS+ subjects.

Only the backend changes. Genuine queries are **full-coverage** re-scans (the whole arch, re-posed
+ sensor noise, keep=1.0), each registered to every gallery arch by BUFFER-X (zero-shot, pretrained
3DMatch, no dental retraining) and scored by the mean nearest-neighbour residual of the aligned
query — the same honest all-query-point residual used by ``eval_bufferx_baseline.residual_bufferx``.
Identity metrics use the **identical** ``eval_id3d.metrics`` definitions as the GICP smoke, so the
two columns are directly comparable. Expected ~1.0 (GICP already saturates full coverage) — the
deliverable is the *complete column*, not a win.

Single-timepoint, synthetic genuine re-scans (same as every headline number) → this is NOT the
gate-#7 longitudinal validation. Writes ``bufferx_identity_full.json``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from train_correspondence import load_norm  # noqa: E402
from eval_bufferx_baseline import _load_bufferx, dense_crop, residual_bufferx  # noqa: E402
from eval_id3d import metrics  # noqa: E402  (IDENTICAL metric defs as the GICP smoke)

OUT = Path(__file__).resolve().parents[1] / "results" / "bufferx_identity_full.json"

DATA = paths.TEETH3DS
N = int(os.environ.get("TP_BUFFERX_N", "40"))          # same 40 real subjects as the GICP smoke
NP = int(os.environ.get("TP_BUFFERX_NP", "8000"))      # dense budget (BUFFER-X does its own FPS)
# crop geometry at keep=1.0 keeps the WHOLE arch either way; "teeth" mirrors the realistic protocol.
MODE = os.environ.get("TP_BUFFERX_FULL_MODE", "teeth")


def main():
    try:
        register = _load_bufferx()
    except ImportError as e:
        print(str(e))
        print("\nAborting: cannot load BUFFER-X here. Wrapper + protocol are wired and ready.")
        sys.exit(1)

    meshes = sorted(Path(DATA).glob("*/*.obj"))[:N]
    if not meshes:
        print(f"No Teeth3DS+ arches under {DATA} (set TP_TEETH3DS). Aborting.")
        sys.exit(1)
    gallery = [g for g in (load_norm(m, NP) for m in meshes) if g is not None]
    qbase = [load_norm(m, NP) for m in meshes]                    # independent sampling for queries
    qbase = [q for q, g in zip(qbase, gallery) if q is not None]
    n = len(gallery)
    gtrees = [cKDTree(g) for g in gallery]
    labels = [str(i) for i in range(n)]                          # each arch = its own subject
    print(f"bufferx FULL-COVERAGE identity (Teeth3DS+, REAL): {n} arches, NP={NP}, "
          f"mode={MODE}, keep=1.0", flush=True)

    t0 = time.time()
    # Genuine query for arch i: full-arch re-pose + sensor noise (keep=1.0). Seed 1000+i echoes the
    # GICP smoke's genuine-query seed convention so the two columns share subjects AND query indices.
    qpts = [dense_crop(qbase[i], np.random.default_rng(1000 + i), 1.0, MODE) for i in range(n)]
    G = np.array([[residual_bufferx(register, qpts[i], gtrees[j]) for j in range(n)]
                  for i in range(n)])
    m = metrics(G, labels)

    out = {
        "dataset": "Teeth3DS+ upper (real intraoral arches, OSF data_part_1, md5-verified)",
        "note": ("FULL-COVERAGE real-arch identity with BUFFER-X (zero-shot, pretrained 3DMatch) as "
                 "the registration/scoring backend — the complete column beside the PCA-init+GICP "
                 "smoke (teeth3ds_identity_smoke_n40.json). Same 40 subjects, identical "
                 "eval_id3d.metrics definitions; genuine = full-arch re-pose (keep=1.0). "
                 "Single-timepoint, synthetic genuine re-scans → NOT gate-#7 longitudinal."),
        "matcher": ("BUFFER-X (arXiv 2503.07940, ICCV 2025) — zero-shot, pretrained 3DMatch, "
                    "no dental retraining"),
        "protocol": "full-coverage self-registration (keep=1.0), Rank-1/Rank-5/EER/AUC gallery ID",
        "n": m["n"], "rank1": m["rank1"], "rank5": m["rank5"], "eer": m["eer"], "auc": m["auc"],
        "dprime": m["dprime"], "genuine_mean": m["genuine_mean"], "impostor_min": m["impostor_min"],
        "n_points": NP, "crop_mode_at_keep1": MODE,
        "gicp_smoke_reference": {"source": "teeth3ds_identity_smoke_n40.json (PCA-init+GICP backend)"},
        "seconds": round(time.time() - t0, 1),
    }
    try:
        gicp = json.loads((OUT.parent / "teeth3ds_identity_smoke_n40.json").read_text())
        out["gicp_smoke_reference"].update({k: gicp.get(k) for k in ("rank1", "rank5", "eer", "auc")})
    except Exception:
        pass

    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  N={m['n']}  Rank-1 {m['rank1']:.3f}  Rank-5 {m['rank5']:.3f}  "
          f"EER {m['eer']:.3f}  AUC {m['auc']:.3f}  d' {m['dprime']:.2f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
