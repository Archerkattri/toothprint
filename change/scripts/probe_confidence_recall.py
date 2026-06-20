#!/usr/bin/env python3
"""Probe: does confidence-gating + heavy TTA make the detector discriminative?

For real DenPAR teeth, measure the detector's bone-level *change* for (a) a stable
re-acquisition and (b) a rendered crestal change, plus a per-tooth confidence
(min spatial-softmax heatmap peak over the bone-level landmarks). If, on the
high-confidence subset, the stable change has small spread and the warp change is
clearly separated, end-to-end detector recall is achievable.
"""
from __future__ import annotations

import argparse
import numpy as np
from PIL import Image

from dcc.data.denpar_adapter import RealDenparAdapter
from dcc.geometry import distance, mean_point
from dcc.landmarks.vitpose_detector import (
    landmark_box, tooth_to_landmarks, ViTPoseLandmarkDetector,
)
from dcc.perturb.image_change import render_crestal_change
from dcc.benchmark.real_image_eval import acquire


def _bonelevel_conf(det, img, tooth, n_tta, rng):
    pts, vis = tooth_to_landmarks(tooth)
    if not (vis[0] and vis[2]):  # need cej_left + crest_mesial for a bone level
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = landmark_box(pts, vis, w, h)
    ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
    if ix2 - ix1 < 2 or iy2 - iy1 < 2:
        return None
    bl_landmarks = [k for k in (0, 1, 2, 3) if vis[k]]
    samples, confs = [], []
    for j in range(n_tta):
        ox, oy = (0, 0) if j == 0 else (int(rng.integers(-4, 5)), int(rng.integers(-4, 5)))
        ax1 = min(max(0, ix1 + ox), w - 2)
        ay1 = min(max(0, iy1 + oy), h - 2)
        ax2 = min(max(ax1 + 2, ix2 + ox), w)
        ay2 = min(max(ay1 + 2, iy2 + oy), h)
        coords, conf = det.predict_crop_conf(img[ay1:ay2, ax1:ax2])
        samples.append([[ax1 + cx, ay1 + cy] for cx, cy in coords])
        confs.append(min(conf[k] for k in bl_landmarks))
    abs_pts = np.median(np.asarray(samples), axis=0)
    cej_mid = mean_point([abs_pts[k] for k in (0, 1) if vis[k]])
    crest_mid = mean_point([abs_pts[k] for k in (2, 3) if vis[k]])
    return distance(cej_mid, crest_mid), float(np.median(confs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/denpar/extracted/Dataset")
    p.add_argument("--weights", default="outputs/vitpose_detector/checkpoint_best.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-teeth", type=int, default=160)
    p.add_argument("--n-tta", type=int, default=15)
    p.add_argument("--delta", type=float, default=30.0)
    p.add_argument("--acq-noise", type=float, default=3.0)
    args = p.parse_args()

    det = ViTPoseLandmarkDetector(args.weights, device=args.device)
    recs = list(RealDenparAdapter(args.data).records("test"))
    rng = np.random.default_rng(0)

    rows = []  # (conf, stable_change, warp_change)
    n = 0
    for rec in recs:
        if n >= args.n_teeth:
            break
        img = np.array(Image.open(rec.image_path).convert("RGB"))
        for tooth in rec.annotation_dict.get("teeth", []):
            if n >= args.n_teeth:
                break
            if not (tooth.get("cej") and tooth.get("crest_line")):
                continue
            base = _bonelevel_conf(det, img, tooth, args.n_tta, rng)
            if base is None:
                continue
            base_bl, conf = base
            stable_img = acquire(img, rng, args.acq_noise)
            st = _bonelevel_conf(det, stable_img, tooth, args.n_tta, rng)
            cej_mid = mean_point(tooth["cej"]); crest_mid = mean_point(tooth["crest_line"])
            warp_img = acquire(render_crestal_change(img, cej_mid, crest_mid, args.delta), rng, args.acq_noise)
            wp = _bonelevel_conf(det, warp_img, tooth, args.n_tta, rng)
            if st is None or wp is None:
                continue
            rows.append((conf, st[0] - base_bl, wp[0] - base_bl))
            n += 1

    rows = np.array(rows)
    conf, stable, warp = rows[:, 0], rows[:, 1], rows[:, 2]
    print(f"\nProbed {len(rows)} teeth (TTA={args.n_tta}, delta={args.delta}px)\n")
    print(f"  confidence range: {conf.min():.3f} .. {conf.max():.3f}  median {np.median(conf):.3f}")
    print(f"\n  {'conf bucket':>22} {'n':>4} {'stable med/IQR':>18} {'warp med/IQR':>18} {'separable?':>10}")
    qs = np.quantile(conf, [0.0, 0.5, 0.75, 0.9])
    for lo, name in [(qs[0], 'all'), (qs[1], 'top50%'), (qs[2], 'top25%'), (qs[3], 'top10%')]:
        m = conf >= lo
        s, wv = stable[m], warp[m]
        s_iqr = np.percentile(s, 75) - np.percentile(s, 25)
        w_iqr = np.percentile(wv, 75) - np.percentile(wv, 25)
        # separable if warp median exceeds stable 90th pct (one-sided)
        sep = np.median(wv) > np.percentile(s, 90)
        print(f"  conf>= {lo:.3f} ({name:>7}) {m.sum():>4} "
              f"{np.median(s):>7.1f}/{s_iqr:>7.1f} {np.median(wv):>7.1f}/{w_iqr:>7.1f} {str(sep):>10}")


if __name__ == "__main__":
    main()
