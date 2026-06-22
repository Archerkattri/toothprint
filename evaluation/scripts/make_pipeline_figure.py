#!/usr/bin/env python3
"""Method schematic for the paper: the unified retrieve -> verify -> certify pipeline with the
accept/abstain decision. A drawn diagram (no data), in the paper palette. Saves docs/pipeline.png."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "docs" / "pipeline.png"
INK, TEAL, AMBER, SLATE, LINE = "#1f2a37", "#0d7d6e", "#b06f1f", "#475569", "#cbd5e1"


def main():
    W, H = 14.3, 3.4
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    bw, bh, ymid = 1.95, 1.12, 1.7
    xs = [0.15, 2.40, 4.65, 6.90, 9.15]
    stages = [
        ("01", "Query", "3D intraoral scan\n2D radiograph"),
        ("02", "Detect", "arch point cloud\nper-tooth landmarks"),
        ("03", "Retrieve", "embedding (DGCNN\n+ ArcFace) → top-$k$"),
        ("04", "Verify", "point correspondence\n→ Procrustes residual"),
        ("05", "Certify", r"split-conformal" "\n" r"threshold $\hat{\tau}_\alpha$"),
    ]

    def box(x, y, w, h, title, sub, fill, edge, tcol, idx=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.13",
                                    linewidth=1.7, edgecolor=edge, facecolor=fill, zorder=3))
        cx = x + w / 2
        ax.text(cx, y + h * 0.66, title, ha="center", va="center", fontsize=13, fontweight="bold", color=tcol, zorder=4)
        ax.text(cx, y + h * 0.30, sub, ha="center", va="center", fontsize=8.6, color=SLATE, zorder=4, linespacing=1.35)
        if idx:
            ax.text(x + 0.16, y + h - 0.16, idx, ha="left", va="top", fontsize=8.5,
                    color=TEAL, fontweight="bold", family="monospace", zorder=4)

    def arrow(x0, y0, x1, y1, color=INK, lw=2.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15,
                                     color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=2))

    for i, (idx, t, s) in enumerate(stages):
        box(xs[i], ymid - bh / 2, bw, bh, t, s, "#ffffff", LINE, INK, idx)
        if i:
            arrow(xs[i - 1] + bw, ymid, xs[i], ymid)

    # decision branch from Certify
    cx_r = xs[-1] + bw
    ow, oh = 2.05, 0.96
    ox = cx_r + 0.55
    box(ox, ymid + 0.55, ow, oh, "Accept", r"subject $\hat{i}$ ·  FMR $\leq \alpha$",
        "#e3f1ee", TEAL, TEAL)
    box(ox, ymid - 0.55 - oh, ow, oh, "Abstain", "partial overlap /\noutside calibration",
        "#f6ecdc", AMBER, AMBER)
    arrow(cx_r, ymid, ox, ymid + 0.55 + oh / 2, color=TEAL)
    arrow(cx_r, ymid, ox, ymid - 0.55 - oh / 2, color=AMBER)
    ax.text(cx_r + 0.30, ymid + 0.92, r"$s\leq\hat{\tau}_\alpha$", fontsize=8.5, color=TEAL, ha="left")
    ax.text(cx_r + 0.30, ymid - 0.92, r"$s>\hat{\tau}_\alpha$", fontsize=8.5, color=AMBER, ha="left")

    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.4)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
