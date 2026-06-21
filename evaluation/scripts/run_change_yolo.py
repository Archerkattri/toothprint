#!/usr/bin/env python3
"""End-to-end bone-level change certificate with YOLO26-pose localization (DenPAR).

The differential sub-pixel registration measurement and the conformal certificate
are identical to the GT/ViTPose runs; only the patch centres (CEJ + bone crest)
come from a fine-tuned full-image YOLO26-pose detector instead of ViTPose crops, so
any recall difference is purely localization quality. Writes
evaluation/results/change_registration_yolo.json (raw per-tooth rows + sweep), the
input to analyze.py and make_change_figure.py.

Runs from the toothprint repo (data/ + the trained weights on the relative paths
below); all imports are standalone (toothprint.bench), the same as eval_change.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from toothprint.bench.benchmark.real_image_eval import acquire
from toothprint.bench.certificate.conformal import AsymmetricConformalInterval, classify_interval
from toothprint.bench.data.denpar_adapter import RealDenparAdapter
from toothprint.bench.geometry import mean_point
from toothprint.bench.landmarks.vitpose_detector import tooth_to_landmarks
from toothprint.bench.perturb.image_change import render_crestal_change
from toothprint.bench.score.registration_change import (
    measure_bonelevel_change_search, snap_to_margin,
)

WEIGHTS = "runs/pose/outputs/yolo_pose/denpar26s/weights/best.pt"
OUT = Path(__file__).resolve().parents[1] / "results" / "change_registration_yolo.json"


def _global_shift(im, sx, sy):
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def _measure_tooth(img, gt_cej, gt_crest, u, ref_c, crest_c, delta, rng, acq, gshift, offsets, snap):
    gx, gy = rng.uniform(-gshift, gshift), rng.uniform(-gshift, gshift)
    warp = render_crestal_change(img, gt_cej.tolist(), gt_crest.tolist(), delta)
    t1 = _global_shift(acquire(warp, rng, acq), gx, gy)
    g1 = cv2.cvtColor(t1, cv2.COLOR_RGB2GRAY)
    g0 = cv2.cvtColor(acquire(img, rng, acq), cv2.COLOR_RGB2GRAY)
    crest = snap_to_margin(g0, tuple(crest_c), tuple(u)) if snap else crest_c
    out = measure_bonelevel_change_search(g0, g1, ref_c, crest, u, offsets, min_response=0.4)
    return None if out is None else out[0]


class YoloLocalizer:
    """Full-image YOLO26-pose CEJ/crest localization, cached per image."""

    def __init__(self, weights, imgsz=960, conf=0.25, cover_all=True):
        self.m = YOLO(weights); self.imgsz = imgsz; self.conf = conf
        self.cover_all = cover_all; self._cache = {}

    def localize(self, img, key, tooth):
        if key not in self._cache:
            r = self.m(img, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
            has = r.keypoints is not None and len(r.keypoints) > 0
            self._cache[key] = (r.keypoints.xy.cpu().numpy() if has else None,
                                r.boxes.xywh.cpu().numpy()[:, :2] if has else None)
        kpts, cent = self._cache[key]
        if kpts is None:
            return None
        pts, vis = tooth_to_landmarks(tooth)
        gc = pts[vis].mean(0)
        j = int(((cent - gc) ** 2).sum(1).argmin())
        # cover_all keeps every tooth with its nearest detection (full coverage, like
        # ViTPose); otherwise drop teeth the detector missed by more than a tooth span.
        if not self.cover_all and \
                np.linalg.norm(cent[j] - gc) > np.linalg.norm(pts[vis].max(0) - pts[vis].min(0)) + 60:
            return None
        k = kpts[j]
        return (k[0] + k[1]) / 2.0, (k[2] + k[3]) / 2.0      # cej centre, crest centre


def iter_yolo(records, yl, *, deltas, rng, acq, gshift, cap, offsets, snap):
    n = 0
    for rec in records:
        if n >= cap:
            break
        img = np.array(Image.open(rec.image_path).convert("RGB"))
        key = str(rec.image_path)
        for tooth in rec.annotation_dict.get("teeth", []):
            if n >= cap:
                break
            if not (tooth.get("cej") and tooth.get("crest_line")):
                continue
            gt_cej = np.array(mean_point(tooth["cej"])); gt_crest = np.array(mean_point(tooth["crest_line"]))
            v = gt_crest - gt_cej; L = float(np.linalg.norm(v))
            if L < 1e-6:
                continue
            u = v / L
            loc = yl.localize(img, key, tooth)
            if loc is None:
                continue
            cej_c, crest_c = np.asarray(loc[0]), np.asarray(loc[1])
            ref_c = cej_c - 0.7 * L * u
            row, ok = {}, True
            for d in deltas:
                m = _measure_tooth(img, gt_cej, gt_crest, u, ref_c, crest_c, d, rng, acq, gshift, offsets, snap)
                if m is None:
                    ok = False; break
                row[d] = m
            if ok:
                yield row; n += 1


def main() -> int:
    p = argparse.ArgumentParser(description="End-to-end change certificate, YOLO26-pose localization")
    p.add_argument("--data", default="data/denpar/extracted/Dataset")
    p.add_argument("--weights", default=WEIGHTS)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.5)            # toothprint operating point
    p.add_argument("--acq-noise", type=float, default=3.0)
    p.add_argument("--global-shift", type=float, default=6.0)
    p.add_argument("--cal-teeth", type=int, default=150)
    p.add_argument("--test-teeth", type=int, default=200)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--detected-only", action="store_true",
                   help="Drop teeth the detector missed instead of full coverage")
    p.add_argument("--snap", action="store_true")
    p.add_argument("--sweep", default="0,4,8,12,16,20,30")
    args = p.parse_args()
    offsets = list(range(-64, 65, 4))

    adapter = RealDenparAdapter(Path(args.data))
    yl = YoloLocalizer(args.weights, conf=args.conf, cover_all=not args.detected_only)
    rng = np.random.default_rng(0)

    predicted, observed = [], []
    for row in iter_yolo(list(adapter.records("train")), yl, deltas=[0.0], rng=rng, acq=args.acq_noise,
                         gshift=args.global_shift, cap=args.cal_teeth, offsets=offsets, snap=args.snap):
        predicted.append(row[0.0]); observed.append(0.0)
    conformal = AsymmetricConformalInterval.fit(predicted, observed, alpha=args.alpha)
    print(f"Localization: YOLO26-pose ({'detected-only' if args.detected_only else 'full coverage'})\n"
          f"Calibrated conformal: q_lo={conformal.q_lo:.3f} q_hi={conformal.q_hi:.3f} px (n={len(predicted)})")

    sweep = [float(x) for x in args.sweep.split(",")]
    test_rows = list(iter_yolo(list(adapter.records("test")), yl, deltas=sweep, rng=rng, acq=args.acq_noise,
                               gshift=args.global_shift, cap=args.test_teeth, offsets=offsets, snap=args.snap))
    print(f"Test teeth: {len(test_rows)}\n  {'change_px':>9} {'recall':>8} {'FPR':>8} {'measured_med':>13}")
    curve = []
    for d in sweep:
        decisions = [classify_interval(conformal.predict(r[d]), tau=args.tau) for r in test_rows]
        prog = sum(x == "progressed" for x in decisions) / len(decisions)
        med = float(np.median([r[d] for r in test_rows]))
        is_change = d >= args.tau
        curve.append({"change_px": d, "certified_change": prog, "measured_median": med,
                      "recall": prog if is_change else None, "fpr": prog if not is_change else None})
        print(f"  {d:>9.0f} {(prog if is_change else float('nan')):>8.3f} "
              f"{(prog if not is_change else float('nan')):>8.3f} {med:>13.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"dataset": "denpar", "localization": "yolo26-pose",
         "coverage": "detected" if args.detected_only else "all", "conf": args.conf,
         "tau": args.tau, "alpha": args.alpha, "acq_noise": args.acq_noise,
         "global_shift": args.global_shift, "n_test_teeth": len(test_rows),
         "q_lo": conformal.q_lo, "q_hi": conformal.q_hi, "sweep": curve,
         "test_rows": [{str(d): r[d] for d in r} for r in test_rows]}, indent=1) + "\n")
    print(f"\n  saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
