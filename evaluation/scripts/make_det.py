#!/usr/bin/env python3
"""Full DET curves (FNMR vs FMR) per identity pillar — the biometric-standard characterization the
dental-ID literature omits (it reports only Rank-N). 3D identity from committed score arrays;
the learned-correspondence partial-overlap pillar recomputed; multimodal IOS from the cached
matrices. Saves docs/det_curves.png + det_curves.json (EER + operating points). Needs the
Poseidon3D arches + corrnet.pt for the CorrNet curve (skipped gracefully if absent)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths

RES = Path(__file__).resolve().parents[1] / "results"
FIG = Path(__file__).resolve().parents[1].parent / "docs" / "det_curves.png"


def det(gen, imp):
    """Distance scores (match if < tau). Returns FMR, FNMR sweep + EER."""
    gen, imp = np.asarray(gen, float), np.asarray(imp, float)
    taus = np.unique(np.concatenate([gen, imp]))
    fmr = np.array([np.mean(imp < t) for t in taus])
    fnmr = np.array([np.mean(gen >= t) for t in taus])
    i = int(np.argmin(np.abs(fmr - fnmr)))
    return fmr, fnmr, float((fmr[i] + fnmr[i]) / 2)


def corrnet_scores():
    """Recompute CorrNet genuine/impostor residuals at keep-0.5 (realistic tooth dropout)."""
    import torch
    from eval_correspondence import WEIGHTS, crop_query, descs, residual
    from train_correspondence import M, load_norm
    from toothprint.identity.embedding import CorrNet
    if not WEIGHTS.exists():
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(WEIGHTS, map_location=dev); net = CorrNet(ck["desc"]).to(dev); net.load_state_dict(ck["net"]); net.eval()
    meshes = sorted(Path(str(paths.POSEIDON3D)).glob("*/*.stl"))
    clouds = [c for c in (load_norm(m, M * 3) for m in meshes) if c is not None]
    held = clouds[ck["n_train"]:]; n = len(held)
    if n < 10:
        return None
    gpts = [h[np.random.default_rng(5 + i).choice(len(h), M, replace=len(h) < M)] for i, h in enumerate(held)]
    gdz = [descs(net, g, dev) for g in gpts]
    qpts = [crop_query(held[i], np.random.default_rng(300 + i), 0.5, "teeth") for i in range(n)]
    qdz = [descs(net, q, dev) for q in qpts]
    G = np.array([[residual(qpts[i], qdz[i], gpts[j], gdz[j]) for j in range(n)] for i in range(n)])
    gen = [G[i, i] for i in range(n)]; imp = [G[i, j] for i in range(n) for j in range(n) if i != j]
    return gen, imp


def main():
    curves = {}
    d3 = json.loads((RES / "id3d.json").read_text())["main"]
    curves["3D identity · GICP · full coverage (N=200)"] = det(d3["genuine"], d3["impostor_round"])
    npz = np.load(RES / "fusion_matrices_n30_0p03_0p8.npz"); IOS = npz["ios"]
    n = len(IOS)
    curves["Multimodal · IOS crowns · hard regime (N=30)"] = det([IOS[i, i] for i in range(n)],
                                                                 [IOS[i, j] for i in range(n) for j in range(n) if i != j])
    cs = corrnet_scores()
    if cs:
        curves["Learned correspondence · 50% tooth loss (N=50)"] = det(*cs)
        print("CorrNet curve computed", flush=True)
    else:
        print("CorrNet curve skipped (no weights/data)", flush=True)

    plt.figure(figsize=(7, 6))
    for label, (fmr, fnmr, eer) in curves.items():
        good = (fmr > 0) & (fnmr > 0)
        plt.loglog(fmr[good] * 100, fnmr[good] * 100, lw=2, label=f"{label}  (EER {eer*100:.1f}%)")
    plt.plot([1e-2, 1e2], [1e-2, 1e2], "k--", lw=0.6, alpha=0.5, label="EER line")
    plt.xlabel("False Match Rate (%)"); plt.ylabel("False Non-Match Rate (%)")
    plt.title("ToothPrint identity — DET curves\n(first full DET for dental identity; literature reports only Rank-N)")
    plt.legend(fontsize=8, loc="lower left"); plt.grid(True, which="both", alpha=0.3)
    plt.xlim(0.05, 100); plt.ylim(0.05, 100); plt.tight_layout()
    FIG.parent.mkdir(exist_ok=True); plt.savefig(FIG, dpi=130)
    out = {k: {"eer": round(v[2], 4)} for k, v in curves.items()}
    (RES / "det_curves.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out.items():
        print(f"  {k}: EER {v['eer']*100:.2f}%", flush=True)
    print(f"saved {FIG}", flush=True)


if __name__ == "__main__":
    main()
