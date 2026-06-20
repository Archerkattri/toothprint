#!/usr/bin/env python3
"""Diagnostic: does the detector track a rendered crestal warp? (no certificate)

For a sample of real scorable teeth, render the crest shift into pixels at several
magnitudes and measure the detector's reported CEJ-to-crest change (with TTA),
no acquisition noise. If median(detected) tracks delta, recall is achievable.
"""
from __future__ import annotations

import argparse
import numpy as np
from PIL import Image

from dcc.data.denpar_adapter import RealDenparAdapter
from dcc.geometry import mean_point
from dcc.landmarks.vitpose_detector import ViTPoseLandmarkDetector, predict_tooth
from dcc.perturb.image_change import render_crestal_change
from dcc.score.periodontal import scalar_change_score


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/denpar/extracted/Dataset")
    p.add_argument("--weights", default="outputs/vitpose_detector/checkpoint_best.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-teeth", type=int, default=80)
    p.add_argument("--n-tta", type=int, default=7)
    p.add_argument("--deltas", default="0,15,30,60,120")
    args = p.parse_args()

    deltas = [float(x) for x in args.deltas.split(",")]
    det = ViTPoseLandmarkDetector(args.weights, device=args.device)
    recs = list(RealDenparAdapter(args.data).records("test"))

    measured = {d: [] for d in deltas}
    n = 0
    rng = np.random.default_rng(0)
    for rec in recs:
        if n >= args.n_teeth:
            break
        img = np.array(Image.open(rec.image_path).convert("RGB"))
        for tooth in rec.annotation_dict.get("teeth", []):
            if n >= args.n_teeth:
                break
            if not (tooth.get("cej") and tooth.get("crest_line")):
                continue
            base = predict_tooth(det, img, tooth, n_tta=args.n_tta, rng=rng)
            if base is None:
                continue
            cej_mid = mean_point(tooth["cej"])
            crest_mid = mean_point(tooth["crest_line"])
            ok = True
            local = {}
            for d in deltas:
                wimg = render_crestal_change(img, cej_mid, crest_mid, d)
                pred = predict_tooth(det, wimg, tooth, n_tta=args.n_tta, rng=rng)
                if pred is None:
                    ok = False
                    break
                try:
                    s = scalar_change_score({"teeth": [base]}, {"teeth": [pred]},
                                            tooth_id=base["tooth_id"])
                except (KeyError, ValueError):
                    ok = False
                    break
                local[d] = s
            if ok:
                for d in deltas:
                    measured[d].append(local[d])
                n += 1

    print(f"\nWarp-detection diagnostic over {n} teeth (TTA={args.n_tta}):")
    print(f"  {'delta_px':>9} {'median':>9} {'IQR':>9} {'p90|abs|':>9}")
    for d in deltas:
        arr = np.array(measured[d])
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
        print(f"  {d:>9.0f} {np.median(arr):>9.2f} {iqr:>9.2f} {np.percentile(np.abs(arr),90):>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
