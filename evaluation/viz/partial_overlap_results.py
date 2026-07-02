#!/usr/bin/env python3
"""One honest chart of the partial-overlap identity picture on **real** dental arches.

Grouped bars, Rank-1 identity at whole-tooth retention keep-1.0 / 0.5 / 0.3, five methods:

  * **BUFFER-X zero-shot** (Teeth3DS+, MEASURED 2026-07-02) — keep-0.5 / keep-0.3 read live
    from ``results/bufferx_baseline.json`` (N=40, **3 crop-seed reps**; whiskers = min–max
    range); keep-1.0 is the **measured** full-coverage Rank-1 from
    ``results/bufferx_identity_full.json`` (BUFFER-X backend on the same protocol as the
    PCA-init+GICP smoke; the rigid pipeline also hits 1.00 full-coverage on Teeth3DS+).
  * **CorrNet** (Poseidon3D, RECORDED) — 0.87 / 0.57 realistic whole-tooth dropout.
  * **rigid GICP** (RECORDED) — 0.23 / 0.10, the family failure mode under 50–70% tooth loss.
  * **DGCNN embedding** (RECORDED) — 0.96 full-coverage (held-out) only; the from-scratch learned
    descriptor, the like-for-like reference for the frozen Sonata head.
  * **Sonata frozen head** (Teeth3DS+, MEASURED 2026-07-02) — the honest NEGATIVE, read live
    from ``results/sonata_identity.json``; a frozen indoor-SSL encoder + ArcFace head does not
    transfer to dental identity in this low-data, head-only recipe.

A method's bar is present only where a measured/recorded number exists; gaps are honest, not
zeros. Every caveat rides in the caption: cross-dataset references (BUFFER-X + Sonata on
Teeth3DS+, CorrNet + GICP + DGCNN recorded on Poseidon3D), N=40 over 3 crop-seed reps (whiskers =
min–max), Sonata = frozen encoder head-only (full fine-tune untested). Palette is the dataviz set, CVD
validated (worst ΔE 16.2). Writes docs/partial_overlap_results.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RES = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parents[2] / "docs" / "partial_overlap_results.png"

# dataviz categorical palette (validated order: aqua, blue, red, orange, violet)
AQUA, BLUE, RED, ORANGE, VIOLET = "#1baf7a", "#2a78d6", "#e34948", "#eb6834", "#4a3aa7"
INK, SEC, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def main():
    bx = json.loads((RES / "bufferx_baseline.json").read_text())["results"]
    so = json.loads((RES / "sonata_identity.json").read_text())["keep_ablation_rank1"]
    dgcnn_full = json.loads((RES / "embedding_identity.json").read_text())["main"]["rank1"]  # 0.96 held-out
    bx_10 = json.loads((RES / "bufferx_identity_full.json").read_text())["rank1"]  # measured full-coverage

    bx_05 = bx["teeth_keep0.5"]["bufferx_rank1"]     # 1.00 (measured, 3 reps)
    bx_03 = bx["teeth_keep0.3"]["bufferx_rank1"]     # 0.95 (measured, 3 reps)
    so_10, so_05, so_03 = so["1.0"], so["0.5"], so["0.3"]  # 0.275 / 0.125 / 0.025 (measured)

    def _err(entry):
        """Asymmetric (v-min, max-v) whisker from the per-rep spread; None if no spread recorded."""
        lo, hi, v = entry.get("min"), entry.get("max"), entry["bufferx_rank1"]
        if lo is None or hi is None or (hi - lo) < 1e-9:
            return None
        return (v - lo, hi - v)
    # measured BUFFER-X whiskers per keep column (keep-1.0 saturates, no spread)
    bx_errs = [None, _err(bx["teeth_keep0.5"]), _err(bx["teeth_keep0.3"])]

    keeps = ["1.0", "0.5", "0.3"]
    NA = np.nan
    # (label, [keep1.0, keep0.5, keep0.3], color, provenance tag, per-keep whisker (lo,hi) or None)
    series = [
        ("BUFFER-X zero-shot",   [bx_10, bx_05, bx_03],     AQUA,   "Teeth3DS+ · measured", bx_errs),
        ("CorrNet",              [NA,   0.87,  0.57],        BLUE,   "Poseidon3D · recorded", None),
        ("rigid GICP",           [NA,   0.23,  0.10],        RED,    "recorded", None),
        ("DGCNN embedding",      [dgcnn_full, NA, NA],       ORANGE, "recorded · full-coverage", None),
        ("Sonata frozen head",   [so_10, so_05, so_03],      VIOLET, "Teeth3DS+ · measured · NEGATIVE", None),
    ]
    legend_tag = {"BUFFER-X zero-shot": "measured", "CorrNet": "recorded", "rigid GICP": "recorded",
                  "DGCNN embedding": "recorded", "Sonata frozen head": "measured"}

    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif",
                         "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": SEC, "ytick.color": SEC})
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    ng, x = len(keeps), np.arange(len(keeps))
    w = 0.15
    off = (np.arange(len(series)) - (len(series) - 1) / 2) * w
    for k, (label, vals, color, _tag, errs) in enumerate(series):
        for gi in range(ng):
            v = vals[gi]
            xpos = x[gi] + off[k]
            if np.isnan(v):
                ax.text(xpos, 0.012, "n/a", ha="center", va="bottom", fontsize=6.5,
                        color=MUTED, rotation=90)
                continue
            ax.bar(xpos, v, w * 0.92, color=color, zorder=3,
                   edgecolor="white", linewidth=0.6)
            e = errs[gi] if errs is not None else None
            if e is not None:                                   # min–max whisker across crop-seed reps
                ax.errorbar(xpos, v, yerr=[[e[0]], [e[1]]], fmt="none", ecolor=INK,
                            elinewidth=1.1, capsize=3, capthick=1.1, zorder=4)
            top = v + (e[1] if e is not None else 0.0)          # label above the whisker cap
            ax.text(xpos, top + 0.018, f"{v:.2f}".lstrip("0") if v < 1 else "1.00",
                    ha="center", va="bottom", fontsize=8, color=INK, fontweight="bold",
                    fontfamily="sans-serif")

    ax.set_xticks(x)
    ax.set_xticklabels(["1.0\n(full arch)", "0.5\n(50% teeth lost)", "0.3\n(70% lost)"])
    ax.set_xlabel("whole-tooth retention  $\\rho$", fontsize=11, color=SEC)
    ax.set_ylabel("Rank-1 identification", fontsize=11, color=SEC)
    ax.set_ylim(0, 1.26)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.set_title("Partial-overlap identity on real dental arches:\n"
                 "zero-shot registration transfers — a frozen foundation-model head does not",
                 fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)

    legend = [Patch(facecolor=c, label=f"{lab}  ({legend_tag[lab]})") for lab, _v, c, _tag, _e in series]
    # legend outside the axes (right) so it never occludes a bar / whisker / value label;
    # savefig(bbox_inches="tight") keeps it in the rendered PNG.
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5,
              framealpha=0.96, edgecolor=GRID, ncol=1, handlelength=1.1, borderpad=0.6,
              labelspacing=0.35)

    cap = ("Real arches, CorrNet whole-tooth-dropout crop protocol.  "
           "BUFFER-X + Sonata measured 2026-07-02 on Teeth3DS+ (N=40; BUFFER-X keep-0.5/0.3 over "
           "3 crop-seed reps, whiskers = min–max range).  "
           f"keep-1.0 BUFFER-X = measured full-coverage (Rank-1 {bx_10:.2f}, bufferx_identity_full.json).  "
           "CorrNet / GICP / DGCNN recorded on Poseidon3D (different real dataset — cross-dataset "
           "comparison).  Sonata = frozen indoor-SSL encoder + ArcFace head only; full fine-tune "
           "untested.  Single-timepoint data — gate #7 (real longitudinal) still open.")
    fig.text(0.012, -0.02, cap, ha="left", va="top", fontsize=7.6, color=MUTED, wrap=True)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
