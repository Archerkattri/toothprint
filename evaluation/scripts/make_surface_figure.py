#!/usr/bin/env python3
"""Surface-certificate evidence figure from the Poseidon3D sweeps.

Left  — recall @ a 1mm change vs reconstruction noise: the raw mean-norm estimator
        collapses once noise rectifies into its conformal radius; the de-biased
        (noise-power-subtracted) estimator holds far longer. False-change stays 0.
Right — the HONEST caveat: the de-biasing gain depends on the noise being
        spatially incoherent. Under correlated (realistic) reconstruction error the
        recall erodes — so the left panel's reach is an upper bound.

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


def main():
    d = json.loads((R / "surface.json").read_text())
    noises = [0.03, 0.05, 0.10, 0.20, 0.40, 0.84]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- Left: recall@1mm vs noise, raw vs de-biased --------------------------
    raw = [rec1(d["baseline_raw"][f"noise_{n}"]) for n in noises]
    deb = [rec1(d["ablations"][f"noise_{n}"]) for n in noises]
    axL.plot(noises, deb, "-o", color="#2ca02c", lw=2.4, ms=7,
             label="de-biased (noise-power subtraction)")
    axL.plot(noises, raw, "-s", color="#d62728", lw=2.2, ms=6,
             label="raw mean-norm (rectifies noise)")
    axL.axvline(0.84, color="#e6a93f", ls="--", lw=1.5)
    axL.annotate("Gaussian-Splatting\nphoto-recon (0.84mm)", xy=(0.84, 0.5),
                 xytext=(0.52, 0.62), fontsize=9, color="#b07d12",
                 arrowprops=dict(arrowstyle="->", color="#e6a93f"))
    axL.axhline(0.0, color="#999", ls=":", lw=1)
    axL.set_title("Recall @ 1 mm change vs reconstruction noise  (false-change = 0)", fontsize=12)
    axL.set_xlabel("reconstruction noise σ (mm)"); axL.set_ylabel("recall")
    axL.set_ylim(-0.03, 1.05); axL.grid(alpha=.3); axL.legend(loc="center left", fontsize=9)
    axL.annotate("de-biasing extends the\nusable range 0.1 → 0.4 mm",
                 xy=(0.4, 1.0), xytext=(0.18, 0.30), fontsize=9, color="#2ca02c",
                 arrowprops=dict(arrowstyle="->", color="#2ca02c"))

    # ---- Right: the honest caveat — gain vs noise correlation -----------------
    corrs = [0.0, 0.5, 0.9, 1.0]
    rc = [rec1(d["correlated"][f"corr_{c}"]) for c in corrs]
    axR.plot(corrs, rc, "-o", color="#1f77b4", lw=2.4, ms=7)
    axR.fill_between(corrs, rc, 0, color="#1f77b4", alpha=0.08)
    axR.set_title("Honest caveat: gain needs incoherent noise  (σ = 0.20 mm)", fontsize=12)
    axR.set_xlabel("spatial correlation of reconstruction noise"); axR.set_ylabel("de-biased recall @ 1 mm")
    axR.set_ylim(-0.03, 1.05); axR.grid(alpha=.3)
    axR.annotate("independent noise:\nfull gain", xy=(0.0, rc[0]), xytext=(0.06, 0.72),
                 fontsize=9, color="#1f77b4", arrowprops=dict(arrowstyle="->", color="#1f77b4"))
    axR.annotate("correlated (realistic):\ngain erodes", xy=(1.0, rc[-1]), xytext=(0.55, 0.55),
                 fontsize=9, color="#555", arrowprops=dict(arrowstyle="->", color="#888"))

    fig.suptitle("ToothPrint — 3D surface-change certificate on real Poseidon3D arches",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for out in [R.parents[1] / "web" / "assets" / "surface_certificate.png",
                R.parents[1] / "docs" / "surface_certificate.png"]:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
