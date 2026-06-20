#!/usr/bin/env python3
"""Generate ViTPose/warp visuals on real DenPAR data (loads the detector once).

Outputs to docs/:
  - dcc_landmark_pred_vs_gt.png   predicted (red) vs GT (green) landmarks on a crop
  - dcc_warp_panel.png            baseline vs pixel-rendered crestal change
  - dcc_warp_detection.png        detected change vs true rendered shift (box plot)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dcc.data.denpar_adapter import RealDenparAdapter
from dcc.geometry import mean_point
from dcc.landmarks.vitpose_detector import (
    landmark_box, predict_tooth, tooth_to_landmarks, ViTPoseLandmarkDetector,
)
from dcc.perturb.image_change import render_crestal_change
from dcc.score.periodontal import scalar_change_score

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"; DOCS.mkdir(exist_ok=True)


def _scorable(rec):
    return [t for t in rec.annotation_dict.get("teeth", []) if t.get("cej") and t.get("crest_line")]


def _crop_box(tooth, w, h):
    pts, vis = tooth_to_landmarks(tooth)
    x1, y1, x2, y2 = landmark_box(pts, vis, w, h)
    return int(x1), int(y1), int(x2), int(y2)


def landmark_overlay(det, recs):
    for rec in recs:
        teeth = _scorable(rec)
        if not teeth:
            continue
        img = np.array(Image.open(rec.image_path).convert("RGB"))
        h, w = img.shape[:2]
        tooth = teeth[len(teeth) // 2]
        x1, y1, x2, y2 = _crop_box(tooth, w, h)
        pred = predict_tooth(det, img, tooth, n_tta=7, rng=np.random.default_rng(0))
        if pred is None:
            continue
        crop = img[y1:y2, x1:x2]
        fig, ax = plt.subplots(figsize=(5.5, 6))
        ax.imshow(crop)
        gx = [p[0] - x1 for p in tooth["cej"] + tooth["crest_line"]]
        gy = [p[1] - y1 for p in tooth["cej"] + tooth["crest_line"]]
        ax.scatter(gx, gy, c="lime", s=90, marker="o", edgecolors="k", label="GT", zorder=3)
        px = [p[0] - x1 for p in pred["cej"] + pred["crest_line"]]
        py = [p[1] - y1 for p in pred["cej"] + pred["crest_line"]]
        ax.scatter(px, py, c="red", s=90, marker="x", linewidths=2.5, label="ViTPose", zorder=4)
        ax.set_title(f"Predicted vs GT landmarks (tooth {tooth.get('tooth_id')})")
        ax.legend(loc="upper right"); ax.axis("off")
        fig.tight_layout(); out = DOCS / "dcc_landmark_pred_vs_gt.png"
        fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)
        return


def warp_panel(det, recs, delta=60.0):
    for rec in recs:
        teeth = _scorable(rec)
        if not teeth:
            continue
        img = np.array(Image.open(rec.image_path).convert("RGB"))
        h, w = img.shape[:2]
        tooth = teeth[len(teeth) // 2]
        x1, y1, x2, y2 = _crop_box(tooth, w, h)
        cej_mid = mean_point(tooth["cej"]); crest_mid = mean_point(tooth["crest_line"])
        warped = render_crestal_change(img, cej_mid, crest_mid, delta)
        diff = np.abs(warped.astype(np.int16) - img.astype(np.int16)).sum(axis=2)

        fig, axes = plt.subplots(1, 3, figsize=(13, 5))
        for ax, im, title in [
            (axes[0], img[y1:y2, x1:x2], "baseline (t0)"),
            (axes[1], warped[y1:y2, x1:x2], f"rendered +{delta:.0f}px crestal change (t1)"),
        ]:
            ax.imshow(im)
            ax.scatter([crest_mid[0] - x1], [crest_mid[1] - y1], c="cyan", s=80, marker="o",
                       edgecolors="k", label="GT crest")
            ax.annotate("", xy=(cej_mid[0] - x1, cej_mid[1] - y1),
                        xytext=(crest_mid[0] - x1, crest_mid[1] - y1),
                        arrowprops=dict(arrowstyle="->", color="yellow", lw=2))
            ax.set_title(title); ax.axis("off")
        axes[2].imshow(diff[y1:y2, x1:x2], cmap="magma")
        axes[2].set_title("|t1 - t0| (the change, in pixels)"); axes[2].axis("off")
        fig.suptitle("The crestal change IS rendered into the follow-up pixels", y=0.98)
        fig.tight_layout(); out = DOCS / "dcc_warp_panel.png"
        fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)
        return


def warp_detection(det, recs, deltas, n_teeth=80, n_tta=7):
    measured = {d: [] for d in deltas}
    rng = np.random.default_rng(0); n = 0
    for rec in recs:
        if n >= n_teeth:
            break
        img = np.array(Image.open(rec.image_path).convert("RGB"))
        for tooth in _scorable(rec):
            if n >= n_teeth:
                break
            base = predict_tooth(det, img, tooth, n_tta=n_tta, rng=rng)
            if base is None:
                continue
            cej_mid = mean_point(tooth["cej"]); crest_mid = mean_point(tooth["crest_line"])
            local, ok = {}, True
            for d in deltas:
                pred = predict_tooth(det, render_crestal_change(img, cej_mid, crest_mid, d),
                                     tooth, n_tta=n_tta, rng=rng)
                if pred is None:
                    ok = False; break
                try:
                    local[d] = scalar_change_score({"teeth": [base]}, {"teeth": [pred]},
                                                   tooth_id=base["tooth_id"])
                except (KeyError, ValueError):
                    ok = False; break
            if ok:
                for d in deltas:
                    measured[d].append(local[d])
                n += 1

    fig, ax = plt.subplots(figsize=(9, 5))
    data = [np.clip(measured[d], -250, 250) for d in deltas]
    ax.boxplot(data, positions=range(len(deltas)), widths=0.6, showfliers=False)
    ax.plot(range(len(deltas)), deltas, "r--o", label="true change (ideal detector)")
    ax.plot(range(len(deltas)), [np.median(measured[d]) for d in deltas],
            "b-s", label="ViTPose median detected")
    ax.set_xticks(range(len(deltas))); ax.set_xticklabels([f"{int(d)}" for d in deltas])
    ax.set_xlabel("true rendered crest shift (px)")
    ax.set_ylabel("detected CEJ-to-crest change (px)")
    ax.set_title(f"A real detector does NOT track a synthetic warp ({n} teeth, TTA={n_tta})\n"
                 "median detected ~0 and non-monotonic — recall needs real longitudinal data")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); out = DOCS / "dcc_warp_detection.png"
    fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/denpar/extracted/Dataset")
    p.add_argument("--weights", default="outputs/vitpose_detector/checkpoint_best.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-teeth", type=int, default=80)
    args = p.parse_args()

    det = ViTPoseLandmarkDetector(args.weights, device=args.device)
    recs = list(RealDenparAdapter(args.data).records("test"))
    landmark_overlay(det, recs)
    warp_panel(det, recs)
    warp_detection(det, recs, [0.0, 15.0, 30.0, 60.0, 120.0], n_teeth=args.n_teeth)


if __name__ == "__main__":
    main()
