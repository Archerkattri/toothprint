#!/usr/bin/env python3
"""Headline results figure (2x2) from committed JSONs — the partial-overlap breakthrough, the
multimodal-fusion complementarity, the dental-work biometric across modalities, and the certified
decision. Pure plotting from result artifacts (no recompute). Saves docs/results_panel.png."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[1] / "results"
FIG = Path(__file__).resolve().parents[1].parent / "docs" / "results_panel.png"
INK, TEAL, AMBER, ROSE, GREEN = "#1f2a37", "#0d9488", "#d97706", "#e11d48", "#16a34a"


def J(name):
    return json.loads((RES / name).read_text())


def main():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cbd5e1", "axes.grid": True,
                         "grid.color": "#eef2f7", "axes.axisbelow": True})
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.5))
    fig.suptitle("ToothPrint — identity results", fontsize=17, fontweight="bold", color=INK, y=0.985)

    # (0,0) partial-overlap breakthrough — grouped bars at keep 0.5 / 0.3
    corr = J("correspondence_identity.json"); ep = J("embedding_partial.json")["keep_ablation"]
    g = {"0.5": 0.23, "0.3": 0.10}
    methods = ["Rigid GICP", "Crop-hardened\nembedding", "CorrNet\n(learned corr.)"]
    k05 = [g["0.5"], ep["0.5"]["crop_hardened_rank1"], corr["results"]["teeth_keep0.5"]["corrnet_rank1"]]
    k03 = [g["0.3"], ep["0.3"]["crop_hardened_rank1"], corr["results"]["teeth_keep0.3"]["corrnet_rank1"]]
    x = np.arange(3); w = 0.38
    a = ax[0, 0]
    a.bar(x - w / 2, k05, w, label="50% teeth lost", color=TEAL)
    a.bar(x + w / 2, k03, w, label="70% teeth lost", color=AMBER)
    for xi, (v1, v2) in enumerate(zip(k05, k03)):
        a.text(xi - w / 2, v1 + 0.02, f"{v1:.2f}", ha="center", fontsize=9, fontweight="bold")
        a.text(xi + w / 2, v2 + 0.02, f"{v2:.2f}", ha="center", fontsize=9, fontweight="bold")
    a.set_xticks(x); a.set_xticklabels(methods); a.set_ylim(0, 1.05); a.set_ylabel("Rank-1 identification")
    a.set_title("Partial-overlap: learned correspondence breaks the ceiling", fontweight="bold", color=INK)
    a.legend(loc="upper left", framealpha=0.9)

    # (0,1) multimodal fusion in the hard regime (complementarity)
    fa = J("fusion_analysis.json")
    names = ["IOS\ncrowns", "CBCT\nbone", "Dental\nwork", "Quality-w.\nfusion", "Oracle\nbound"]
    vals = [fa["ios_crowns"]["rank1"], fa["cbct_bone"]["rank1"], fa["dental_work"]["rank1"],
            fa["qweighted_all_three"]["rank1"], fa["oracle_rank1_any_modality"]]
    cols = [TEAL, TEAL, TEAL, GREEN, INK]
    a = ax[0, 1]
    a.bar(range(5), vals, color=cols)
    for i, v in enumerate(vals):
        a.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    a.axhline(max(vals[:3]), ls="--", lw=1, color=ROSE, alpha=0.7)
    a.set_xticks(range(5)); a.set_xticklabels(names); a.set_ylim(0, 1.08); a.set_ylabel("Rank-1 (hard regime, N=30)")
    a.set_title("Multimodal fusion: gains with the right estimator", fontweight="bold", color=INK)

    # (1,0) dental-work biometric across modalities + robustness
    dw = J("dentalwork_2d.json")["robustness_ablation"]; mm = J("multimodal_full.json")
    labels = ["CBCT\n(HU>2500)\nn=55", "2D radiograph\neasy\nn=165", "2D radiograph\nhard\n(jitter+drop)", "2D radiograph\nharder"]
    vals = [mm["dental_work"]["rank1"], dw["easy(jit0.02)"]["rank1"], dw["hard(jit0.05+drop1)"]["rank1"], dw["harder(jit0.10+drop1)"]["rank1"]]
    a = ax[1, 0]
    a.bar(range(4), vals, color=[INK, TEAL, TEAL, TEAL])
    for i, v in enumerate(vals):
        a.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    a.set_xticks(range(4)); a.set_xticklabels(labels, fontsize=8.5); a.set_ylim(0, 1.08)
    a.set_ylabel("Rank-1 (restoration pattern)")
    a.set_title("Dental-work biometric: works on CBCT and plain radiographs", fontweight="bold", color=INK)

    # (1,1) the certified decision — full vs partial (abstain)
    uni = J("unified_identity.json"); det = J("det_curves.json")
    a = ax[1, 1]; a.axis("off")
    e3 = next(v["eer"] for k, v in det.items() if "3D identity" in k)
    rows = [("Full-coverage identity (N=200)", f"Rank-1 0.995 · EER {e3*100:.1f}%"),
            ("Conformal certificate", "empirical FMR ≤ α, distribution-free"),
            ("Unified decision · full coverage", f"FNIR@FMR=1%  =  {uni['coverage']['1.0']['unified_fnir_at_fmr1pct']:.2f}"),
            ("Unified decision · 50% tooth loss", f"FNIR@FMR=1%  =  {uni['coverage']['0.5']['unified_fnir_at_fmr1pct']:.2f}  →  ABSTAIN"),
            ("Cross-dataset (Teeth3DS, learned)", "0.87 → 0.42 — honest domain gap"),
            ("The one open gate", "real cross-session data (#7)")]
    a.text(0.5, 1.02, "The certified accept / abstain decision", ha="center", fontsize=11.5,
           fontweight="bold", color=INK, transform=a.transAxes)
    for i, (k, v) in enumerate(rows):
        y = 0.86 - i * 0.155
        a.text(0.02, y, k, fontsize=10, color=INK, transform=a.transAxes)
        a.text(0.98, y, v, fontsize=9.5, color=TEAL, ha="right", fontweight="bold", transform=a.transAxes)
        a.plot([0.02, 0.98], [y - 0.05, y - 0.05], color="#eef2f7", lw=1, transform=a.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.97]); FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=140, facecolor="white"); print(f"saved {FIG}")


if __name__ == "__main__":
    main()
