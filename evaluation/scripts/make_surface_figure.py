#!/usr/bin/env python3
"""Surface-certificate evidence figure from the Poseidon3D sweeps (3 panels).

A — global uniform change, recall@1mm vs reconstruction noise: the raw mean-norm
    estimator collapses once noise rectifies into its conformal radius; the
    de-biased (noise-power-subtracted) estimator holds far longer.
B — LOCALIZED (realistic) patch change: a whole-surface average dilutes it to
    nothing; the per-region max statistic recovers the undiluted signal.
C — localized change under correlated (realistic) reconstruction noise: the
    regional detector stays usable for larger changes; the honest residual is tiny
    changes under heavy correlation. False-change is 0 in every panel.

Reads evaluation/results/surface.json (committed) — reproducible, no GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results"


def rec1(block):
    return next(x["changed_rate"] for x in block["curve"] if x["change_mm"] == 1.0)


def curve(block):
    xs = [x["change_mm"] for x in block["curve"]]
    ys = [x["changed_rate"] for x in block["curve"]]
    return xs, ys


def main():
    d = json.loads((R / "surface.json").read_text())
    noises = [0.03, 0.05, 0.10, 0.20, 0.40, 0.84]
    loc = d["localized"]

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 5.0))

    # A — de-biasing extends the usable noise range (global uniform change) ------
    raw = [rec1(d["baseline_raw"][f"noise_{n}"]) for n in noises]
    deb = [rec1(d["ablations"][f"noise_{n}"]) for n in noises]
    axA.plot(noises, deb, "-o", color="#2ca02c", lw=2.4, ms=6, label="de-biased")
    axA.plot(noises, raw, "-s", color="#d62728", lw=2.0, ms=5, label="raw mean-norm")
    axA.axvline(0.84, color="#e6a93f", ls="--", lw=1.4)
    axA.annotate("photo-recon\n0.84 mm", xy=(0.84, 0.45), xytext=(0.5, 0.6), fontsize=8,
                 color="#b07d12", arrowprops=dict(arrowstyle="->", color="#e6a93f"))
    axA.set_title("A · De-bias extends usable noise\n(global change)", fontsize=11)
    axA.set_xlabel("reconstruction noise σ (mm)"); axA.set_ylabel("recall @ 1 mm")
    axA.set_ylim(-0.03, 1.05); axA.grid(alpha=.3); axA.legend(loc="center left", fontsize=8)

    # B — regional recovers a LOCALIZED change global dilutes (σ=0.2, indep) -----
    xg, yg = curve(loc["global_corr0.0"]); xr, yr = curve(loc["regional_corr0.0"])
    axB.plot(xr, yr, "-o", color="#1f77b4", lw=2.4, ms=6, label="regional (max over regions)")
    axB.plot(xg, yg, "-s", color="#888", lw=2.0, ms=5, label="global (whole-surface avg)")
    axB.axvline(0.75, color="#bbb", ls=":", lw=1)
    axB.set_title("B · Localized patch change\n(σ=0.2 mm, incoherent)", fontsize=11)
    axB.set_xlabel("localized change magnitude (mm)"); axB.set_ylabel("recall")
    axB.set_ylim(-0.03, 1.05); axB.grid(alpha=.3); axB.legend(loc="center right", fontsize=8)
    axB.annotate("global dilutes a patch\nchange to nothing", xy=(1.5, 0.02), xytext=(0.9, 0.30),
                 fontsize=8, color="#666", arrowprops=dict(arrowstyle="->", color="#888"))

    # C — regional under correlated noise: honest residual ----------------------
    x0, y0 = curve(loc["regional_corr0.0"]); x9, y9 = curve(loc["regional_corr0.9"])
    axC.plot(x0, y0, "-o", color="#1f77b4", lw=2.4, ms=6, label="incoherent noise")
    axC.plot(x9, y9, "-^", color="#9467bd", lw=2.2, ms=6, label="correlated (corr 0.9)")
    axC.set_title("C · Regional vs correlated noise\n(honest residual)", fontsize=11)
    axC.set_xlabel("localized change magnitude (mm)"); axC.set_ylabel("regional recall")
    axC.set_ylim(-0.03, 1.05); axC.grid(alpha=.3); axC.legend(loc="center right", fontsize=8)
    axC.annotate("correlated noise still\ncosts small changes", xy=(1.0, y9[5]), xytext=(1.1, 0.62),
                 fontsize=8, color="#6a4ca0", arrowprops=dict(arrowstyle="->", color="#9467bd"))

    fig.suptitle("ToothPrint — 3D surface-change certificate on real Poseidon3D arches "
                 "(false-change = 0 in every panel)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for out in [R.parents[1] / "web" / "assets" / "surface_certificate.png",
                R.parents[1] / "docs" / "surface_certificate_v2.png"]:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
