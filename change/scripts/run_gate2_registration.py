#!/usr/bin/env python3
"""Gate-2 change certificate via sub-pixel registration on real DenPAR images.

Replaces independent per-timepoint landmark regression (which compounds the
detector's ~35px error into ~100px of bone-level noise) with a *differential*
registration measurement (dcc/score/registration_change.py): the bone-margin
patch is template-matched between t0 and t1, CEJ-referenced for motion
robustness. This drives the stable-pair measurement noise to ~0.1px, so the
conformal certificate certifies sub-mm change at a ~0% false-progression rate.

Localization of the patch centres is either the GT landmarks (the
measurement-precision ceiling) or the trained ViTPose detector (fully
end-to-end). Run both to separate measurement quality from localization.

    python scripts/run_gate2_registration.py --data data/denpar/extracted/Dataset \
        --output outputs/gate2_registration --tau 6
    python scripts/run_gate2_registration.py ... --detector   # end-to-end
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dcc.certificate.conformal import AsymmetricConformalInterval, classify_interval
from dcc.data.denpar_adapter import RealDenparAdapter
from dcc.geometry import mean_point
from dcc.perturb.image_change import render_crestal_change
from dcc.benchmark.real_image_eval import acquire
from dcc.score.registration_change import (
    measure_bonelevel_change, measure_bonelevel_change_search,
)


def _global_shift(im, sx, sy):
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def _measure_tooth(img, gt_cej, gt_crest, u, ref_c, crest_c, delta, rng, acq, gshift, offsets):
    gx, gy = rng.uniform(-gshift, gshift), rng.uniform(-gshift, gshift)
    warp = render_crestal_change(img, gt_cej.tolist(), gt_crest.tolist(), delta)
    t1 = _global_shift(acquire(warp, rng, acq), gx, gy)
    g1 = cv2.cvtColor(t1, cv2.COLOR_RGB2GRAY)
    g0 = cv2.cvtColor(acquire(img, rng, acq), cv2.COLOR_RGB2GRAY)
    if offsets is None:
        out = measure_bonelevel_change(g0, g1, ref_c, crest_c, u)
    else:
        out = measure_bonelevel_change_search(g0, g1, ref_c, crest_c, u, offsets)
    return None if out is None else out[0]


def _iter_measurements(records, det, *, deltas, rng, acq, gshift, cap, offsets):
    """Yield dict(true_change -> measured) per tooth (one realization each)."""
    n = 0
    for rec in records:
        if n >= cap:
            break
        img = np.array(Image.open(rec.image_path).convert("RGB"))
        for tooth in rec.annotation_dict.get("teeth", []):
            if n >= cap:
                break
            if not (tooth.get("cej") and tooth.get("crest_line")):
                continue
            gt_cej = np.array(mean_point(tooth["cej"]))
            gt_crest = np.array(mean_point(tooth["crest_line"]))
            v = gt_crest - gt_cej
            L = float(np.linalg.norm(v))
            if L < 1e-6:
                continue
            u = v / L
            if det is not None:
                from dcc.landmarks.vitpose_detector import predict_tooth
                pred = predict_tooth(det, img, tooth, n_tta=9, rng=rng)
                if pred is None or not pred.get("cej") or not pred.get("crest_line"):
                    continue
                cej_c = np.array(mean_point(pred["cej"]))
                crest_c = np.array(mean_point(pred["crest_line"]))
            else:
                cej_c, crest_c = gt_cej, gt_crest
            # Reference patch on the crown, coronal to the CEJ and well outside the
            # crestal warp support, so global motion cancels without subtracting
            # any of the real crest signal.
            ref_c = cej_c - 0.4 * L * u
            row = {}
            ok = True
            for d in deltas:
                m = _measure_tooth(img, gt_cej, gt_crest, u, ref_c, crest_c, d, rng, acq, gshift, offsets)
                if m is None:
                    ok = False
                    break
                row[d] = m
            if ok:
                yield row
                n += 1


def main() -> int:
    p = argparse.ArgumentParser(description="Gate-2 registration-based certificate (DenPAR)")
    p.add_argument("--data", default="data/denpar/extracted/Dataset")
    p.add_argument("--output", default="outputs/gate2_registration")
    p.add_argument("--detector", action="store_true", help="ViTPose localization (end-to-end)")
    p.add_argument("--weights", default="outputs/vitpose_detector/checkpoint_best.pt")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=6.0, help="Clinically significant change (px)")
    p.add_argument("--acq-noise", type=float, default=3.0)
    p.add_argument("--global-shift", type=float, default=6.0, help="Acquisition repositioning (px)")
    p.add_argument("--cal-teeth", type=int, default=150)
    p.add_argument("--test-teeth", type=int, default=200)
    p.add_argument("--cal-delta", type=float, default=20.0)
    p.add_argument("--sweep", default="0,4,8,12,16,20,30")
    p.add_argument("--multipatch", action="store_true",
                   help="Search candidate crest patches and take the max — finds the "
                        "moving margin when localization is coarse (default on with --detector)")
    args = p.parse_args()
    # Coarse detector localization needs the candidate search to find the margin.
    offsets = list(range(-40, 41, 8)) if (args.multipatch or args.detector) else None

    root = Path(args.data)
    if not root.exists():
        print(f"ERROR: DenPAR dataset not found at {root}", file=sys.stderr)
        return 1

    det = None
    if args.detector:
        from dcc.landmarks.vitpose_detector import ViTPoseLandmarkDetector
        det = ViTPoseLandmarkDetector(args.weights, device="cuda")

    adapter = RealDenparAdapter(root)
    rng = np.random.default_rng(0)

    # Calibrate the null on STABLE pairs only: the certificate is a one-sided
    # conformal test against "no change", so FPR <= alpha by construction and the
    # interval is not inflated by the (gain<1) attenuation of a calibration change.
    cal_deltas = [0.0]
    predicted, observed = [], []
    for row in _iter_measurements(list(adapter.records("train")), det,
                                  deltas=cal_deltas, rng=rng, acq=args.acq_noise,
                                  gshift=args.global_shift, cap=args.cal_teeth, offsets=offsets):
        for d in cal_deltas:
            predicted.append(row[d]); observed.append(d)
    conformal = AsymmetricConformalInterval.fit(predicted, observed, alpha=args.alpha)
    loc = "ViTPose (end-to-end)" if det is not None else "GT (measurement ceiling)"
    print(f"Localization: {loc}")
    print(f"Calibrated conformal: q_lo={conformal.q_lo:.3f} q_hi={conformal.q_hi:.3f} px "
          f"(n={len(predicted)})")

    # Test sweep.
    sweep = [float(x) for x in args.sweep.split(",")]
    test_rows = list(_iter_measurements(list(adapter.records("test")), det,
                                        deltas=sweep, rng=rng, acq=args.acq_noise,
                                        gshift=args.global_shift, cap=args.test_teeth, offsets=offsets))
    print(f"Test teeth: {len(test_rows)}\n")
    print(f"  {'change_px':>9} {'recall':>8} {'FPR':>8} {'stable_cert':>12} {'measured_med':>13}")
    curve = []
    for d in sweep:
        decisions = [classify_interval(conformal.predict(r[d]), tau=args.tau) for r in test_rows]
        prog = sum(x == "progressed" for x in decisions) / len(decisions)
        stab = sum(x == "stable" for x in decisions) / len(decisions)
        med = float(np.median([r[d] for r in test_rows]))
        is_change = d >= args.tau
        curve.append({"change_px": d, "certified_change": prog, "stable_cert": stab,
                      "measured_median": med,
                      "recall": prog if is_change else None,
                      "fpr": prog if not is_change else None})
        rc = prog if is_change else float("nan")
        fp = prog if not is_change else float("nan")
        print(f"  {d:>9.0f} {rc:>8.3f} {fp:>8.3f} {stab:>12.3f} {med:>13.2f}")

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": "denpar", "localization": "vitpose" if det is not None else "gt",
               "tau": args.tau, "alpha": args.alpha, "acq_noise": args.acq_noise,
               "global_shift": args.global_shift, "n_test_teeth": len(test_rows),
               "q_lo": conformal.q_lo, "q_hi": conformal.q_hi, "sweep": curve}
    suffix = "detector" if det is not None else "gt"
    (out_dir / f"metrics_{suffix}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n  Metrics: {out_dir / f'metrics_{suffix}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
