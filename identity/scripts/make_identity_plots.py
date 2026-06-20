#!/usr/bin/env python3
"""Visualize dental identification: genuine vs impostor match distributions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ID = Path(__file__).resolve().parents[1]
DOCS = ID.parent / "docs"; DOCS.mkdir(exist_ok=True)


def _split(metrics_path):
    d = json.loads(Path(metrics_path).read_text())
    key = "rmse" if "rmse" in d else "residual"
    M = np.array(d[key]); labels = d["labels"]
    gen, imp = [], []
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            (gen if li == lj else imp).append(M[i, j])
    return np.array(gen), np.array(imp), d


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panels = [
        (ID / "outputs/mesh_identification/metrics.json", "3D intraoral-scan arches", "RMSE (mm)", axes[0]),
        (ID / "outputs/landmark_identification/metrics.json", "2D radiograph constellations", "residual (px)", axes[1]),
    ]
    for path, title, xlabel, ax in panels:
        if not path.exists():
            ax.set_title(f"{title}\n(no data)"); continue
        gen, imp, d = _split(path)
        m = identification_rank1(d)
        hi = np.percentile(np.concatenate([gen, imp]), 99)
        bins = np.linspace(0, hi, 40)
        ax.hist(gen, bins=bins, alpha=0.7, color="#2ca02c", label=f"genuine (same person), n={len(gen)}")
        ax.hist(imp, bins=bins, alpha=0.6, color="#d62728", label=f"impostor (different), n={len(imp)}")
        ax.axvline(gen.max(), color="#2ca02c", ls="--", lw=1)
        ax.axvline(imp.min(), color="#d62728", ls="--", lw=1)
        ax.set_xlabel(xlabel); ax.set_ylabel("count")
        ax.set_title(f"{title}\nRank-1 = {m['rank1_accuracy']:.3f}   d' = {m['decidability_dprime']:.1f}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("ToothPrint — recognising a person by their teeth (genuine vs impostor)", y=1.00)
    fig.tight_layout()
    out = DOCS / "toothprint_identification.png"
    fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


def identification_rank1(d):
    return {"rank1_accuracy": d["rank1_accuracy"], "decidability_dprime": d["decidability_dprime"]}


if __name__ == "__main__":
    main()
