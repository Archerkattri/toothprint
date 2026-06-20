#!/usr/bin/env python3
"""Repositioning-robustness figure: single-reference vs multi-anchor affine.

On real DenPAR teeth with NO real bone change, the stable-pair |measured change|
should be 0. A single reference patch cancels only translation, so rotation +
projection magnification leak straight into the measurement (and the conformal
radius). The multi-anchor affine model cancels them. Reads
evaluation/results/change_repositioning.json (committed) — reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results"


def main():
    d = json.loads((R / "change_repositioning.json").read_text())
    sev = d["severities"]
    x = list(range(len(sev)))
    labels = [f"{s['rotation_deg']:.0f}°\n{int((s['magnification']-1)*100)}%\n{s['translation_px']:.0f}px" for s in sev]
    s_med = [s["single_median"] for s in sev]
    s_p90 = [s["single_p90"] for s in sev]
    a_med = [s["anchored_median"] for s in sev]
    a_p90 = [s["anchored_p90"] for s in sev]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(x, s_med, "-s", color="#d62728", lw=2.4, ms=7, label="single reference (median)")
    ax.fill_between(x, s_med, s_p90, color="#d62728", alpha=0.12, label="single ref (→p90)")
    ax.plot(x, a_med, "-o", color="#2ca02c", lw=2.4, ms=7, label="multi-anchor affine (median)")
    ax.fill_between(x, a_med, a_p90, color="#2ca02c", alpha=0.15, label="anchored (→p90)")
    ax.axhline(6.0, color="#888", ls=":", lw=1.2)
    ax.annotate("≈0.6 mm — a spurious 'progression'", xy=(0.5, 6.0), xytext=(0.6, 9.5),
                fontsize=9, color="#555", arrowprops=dict(arrowstyle="->", color="#888"))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("repositioning between visits  (rotation / magnification / translation)")
    ax.set_ylabel("stable-pair |measured change| (px)  — ideal = 0")
    ax.set_title("Change measurement is robust to repositioning, not just translation\n"
                 "(real DenPAR teeth, no real bone change)", fontsize=12)
    ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    for out in [R.parents[1] / "docs" / "repositioning_robustness_v2.png",
                R.parents[1] / "web" / "assets" / "repositioning_robustness.png"]:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
