#!/usr/bin/env python3
"""Benchmark **BUFFER-X** zero-shot registration against our CorrNet on the SAME partial-overlap
identity protocol (#1), so the numbers are directly comparable.

BUFFER-X (Kim et al., *ICCV 2025*, arXiv 2503.07940; code https://github.com/MIT-SPARK/BUFFER-X)
is a zero-shot point-cloud registration method — a generalist correspondence + pose estimator
tuned for cross-domain transfer. Our custom CorrNet (toothprint.identity.embedding.CorrNet) is a
dental-specialised per-point descriptor. This script runs BUFFER-X on the **identical** protocol
as eval_correspondence.py — held-out unseen Poseidon3D subjects, keep-0.5 / keep-0.3 crops
(planar + realistic discrete whole-tooth dropout), Rank-1 gallery identification — and writes
bufferx_baseline.json next to correspondence_identity.json for a head-to-head table.

CorrNet reference numbers to beat (from correspondence_identity.json, `teeth` = realistic
dropout mode):

    protocol            CorrNet Rank-1
    ------------------  --------------
    planar  keep-0.5    ~0.87
    planar  keep-0.3    (see json)
    teeth   keep-0.5    ~0.87   (realistic dropout)
    teeth   keep-0.3    ~0.57   (realistic dropout)

    (baselines for context: crop-hardened embedding ~0.64 @ keep-0.5, rigid GICP ~0.23 @ keep-0.5)

BUFFER-X is **lazy-imported** with a documented install; if it (or its weights) is unavailable
the script prints the exact setup steps and exits without fabricating numbers. It reuses the crop
generator and Rank-1 scorer from eval_correspondence.py so the two evaluations are byte-identical
except for the matcher under test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_correspondence import DATA, M, load_norm, rot  # noqa: E402
from eval_correspondence import KEEP, crop_query, rank1_auc  # noqa: E402  (same protocol)

OUT = Path(__file__).resolve().parents[1] / "results" / "bufferx_baseline.json"
REPS = 3

_INSTALL_HINT = (
    "BUFFER-X is not importable. Install (GPU, ~CUDA 11.8/12.x) with:\n"
    "  git clone https://github.com/MIT-SPARK/BUFFER-X\n"
    "  cd BUFFER-X && pip install -r requirements.txt && pip install -e .\n"
    "  # fetch the released generalist checkpoint per the repo README (weights/ dir)\n"
    "Then set BUFFERX_CKPT to the checkpoint path. See evaluation/scripts/RUN.md.\n"
)


def _load_bufferx():
    """Lazy import + build the pretrained BUFFER-X registrar. Returns a callable
    ``register(src (Ns,3), dst (Nd,3)) -> (R (3,3), t (3,), inlier_score float)`` or raises
    ImportError with install instructions. The exact symbol names track the upstream repo; adapt
    here if the API drifts — this is the single integration point."""
    import os

    try:
        import bufferx  # type: ignore  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the dep
        raise ImportError(_INSTALL_HINT) from e
    ckpt = os.environ.get("BUFFERX_CKPT")
    if not ckpt or not Path(ckpt).exists():
        raise ImportError(_INSTALL_HINT + f"\n(BUFFERX_CKPT={ckpt!r} not found)")
    from bufferx.api import load_model, register  # type: ignore

    model = load_model(ckpt)

    def _reg(src, dst):
        R, t, score = register(model, src.astype(np.float32), dst.astype(np.float32))
        return np.asarray(R), np.asarray(t), float(score)

    return _reg


def residual_bufferx(register, qpts, gpts):
    """Score a query crop against a gallery arch: register, then mean point-to-alignment residual
    (lower = better match), mirroring eval_correspondence.residual's honest all-query-point metric."""
    R, t, _score = register(qpts, gpts)
    aligned = qpts @ R.T + t
    # nearest-neighbour residual of every aligned query point to the gallery cloud
    d = np.sqrt(((aligned[:, None, :] - gpts[None, :, :]) ** 2).sum(-1))
    return float(d.min(1).mean())


def main():
    try:
        register = _load_bufferx()
    except ImportError as e:
        print(str(e))
        print("\nAborting: cannot run BUFFER-X here. Wrapper + protocol are wired and ready; "
              "run on a GPU box with the install above to produce bufferx_baseline.json.")
        sys.exit(1)

    meshes = sorted(Path(DATA).glob("*/*.stl"))
    clouds = [c for c in (load_norm(m, M * 3) for m in meshes) if c is not None]
    # Match eval_correspondence's held-out split (uses corrnet.pt n_train; default 150).
    n_train = 150
    held = clouds[n_train:]; n = len(held)
    print(f"bufferx eval: {n} held-out unseen subjects ...", flush=True)
    gpts = [h[np.random.default_rng(5 + i).choice(len(h), M, replace=len(h) < M)]
            for i, h in enumerate(held)]

    res = {"n_heldout": n, "reps": REPS, "matcher": "BUFFER-X (arXiv 2503.07940, ICCV 2025)",
           "protocol": "identical to eval_correspondence.py",
           "corrnet_reference": {"teeth_keep0.5": 0.87, "teeth_keep0.3": 0.57,
                                 "note": "authoritative values live in correspondence_identity.json"},
           "results": {}}
    for mode in ("planar", "teeth"):
        for keep in (0.5, 0.3):
            r1s, aucs = [], []
            for r in range(REPS):
                qpts = [crop_query(held[i], np.random.default_rng(300 + 91 * r + i), keep, mode)
                        for i in range(n)]
                G = np.array([[residual_bufferx(register, qpts[i], gpts[j]) for j in range(n)]
                              for i in range(n)])
                r1, auc = rank1_auc(G); r1s.append(r1); aucs.append(auc)
            res["results"][f"{mode}_keep{keep}"] = {"bufferx_rank1": round(float(np.mean(r1s)), 3),
                                                     "std": round(float(np.std(r1s)), 3),
                                                     "auc": round(float(np.mean(aucs)), 3)}
            print(f"  {mode:6s} keep {keep}: BUFFER-X Rank-1 {np.mean(r1s):.3f}±{np.std(r1s):.3f}  "
                  f"AUC {np.mean(aucs):.3f}", flush=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
