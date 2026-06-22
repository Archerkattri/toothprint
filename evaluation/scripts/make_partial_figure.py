#!/usr/bin/env python3
"""Partial-overlap figure for the paper: Rank-1 vs whole-tooth retention for rigid GICP, the
embedding baselines, and learned correspondence (CorrNet). Pure plotting from committed JSONs
(embedding_partial.json, correspondence_identity.json); rigid GICP is the reference value used
throughout the paper (Table 2). Saves docs/partial_overlap.png."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parents[2] / "docs" / "partial_overlap.png"
INK, TEAL, AMBER, ROSE, GRAY = "#1f2a37", "#0d9488", "#d97706", "#e11d48", "#94a3b8"


def main():
    ep = json.loads((RES / "embedding_partial.json").read_text())["keep_ablation"]
    co = json.loads((RES / "correspondence_identity.json").read_text())["results"]
    keeps = ["1.0", "0.5", "0.3"]
    x = np.arange(3)

    rigid = [1.00, 0.23, 0.10]                                  # GICP reference (Table 2)
    base = [ep[k]["baseline_rank1"] for k in keeps]
    crop = [ep[k]["crop_hardened_rank1"] for k in keeps]
    corr = [np.nan, co["teeth_keep0.5"]["corrnet_rank1"], co["teeth_keep0.3"]["corrnet_rank1"]]

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#cbd5e1",
                         "axes.grid": True, "grid.color": "#eef2f7", "axes.axisbelow": True})
    fig, a = plt.subplots(figsize=(6.6, 4.7))
    series = [("Rigid GICP", rigid, ROSE, "o", "-"),
              ("Embedding baseline", base, GRAY, "s", "--"),
              ("Crop-hardened embedding", crop, AMBER, "^", "-"),
              ("CorrNet (learned correspondence)", corr, TEAL, "D", "-")]
    for name, y, c, m, ls in series:
        a.plot(x, y, ls, marker=m, color=c, lw=2.4, ms=8, label=name)
    # annotate the recovery at 50% loss
    a.annotate("", xy=(1, corr[1]), xytext=(1, rigid[1]),
               arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    a.text(1.06, (corr[1] + rigid[1]) / 2, "3.8×\nrecovery", fontsize=9.5,
           color=INK, va="center", fontweight="bold")
    for xi, v in zip(x, rigid):
        a.text(xi, v - 0.06, f"{v:.2f}", ha="center", fontsize=8.5, color=ROSE)
    for xi, v in list(zip(x, corr))[1:]:
        a.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=8.5, color=TEAL, fontweight="bold")

    a.set_xticks(x)
    a.set_xticklabels(["1.0\n(full arch)", "0.5\n(50% teeth lost)", "0.3\n(70% lost)"])
    a.set_xlabel("whole-tooth retention $\\rho$")
    a.set_ylabel("Rank-1 identification (held-out subjects)")
    a.set_ylim(0, 1.06)
    a.set_title("Partial overlap: rigid matching collapses, learned\ncorrespondence recovers the identity signal",
                fontsize=12.5, fontweight="bold", color=INK)
    a.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
