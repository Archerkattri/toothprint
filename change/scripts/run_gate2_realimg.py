#!/usr/bin/env python3
"""Gate-2 with MEASURABLE detector recall on real image pairs (DenPAR).

This is the root-fixed recall evaluation. Instead of injecting the crestal change
into the annotation only (invisible to a pixel-reading detector), the follow-up
image has the bone crest **rendered moved** in pixels. ViTPose is run on both
timepoint images and the change is the difference of the two predicted CEJ-to-
crest distances, so recall and false-progression rate are real, detector-driven
numbers — no synthetic stand-in, no fallback.

    python scripts/run_gate2_realimg.py \
        --data data/denpar/extracted/Dataset \
        --output outputs/gate2_realimg \
        --detector-weights outputs/vitpose_detector/checkpoint_best.pt \
        --detector-device cuda --delta-px 30 --tau 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dcc.certificate.conformal import AsymmetricConformalInterval
from dcc.data.denpar_adapter import RealDenparAdapter
from dcc.benchmark.real_image_eval import (
    evaluate_real_image_pairs,
    summarize_real_image_decisions,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Gate-2 real-image detector recall (DenPAR)")
    p.add_argument("--data", default="data/denpar/extracted/Dataset")
    p.add_argument("--output", default="outputs/gate2_realimg")
    p.add_argument("--detector-weights", required=True,
                   help="Trained ViTPose checkpoint (.pt) from train_vitpose_detector.py")
    p.add_argument("--detector-device", default="auto")
    p.add_argument("--delta-px", type=float, default=30.0,
                   help="True crestal bone-loss rendered into follow-up pixels (px)")
    p.add_argument("--tau", type=float, default=15.0,
                   help="Clinically meaningful change threshold (px) for certification")
    p.add_argument("--alpha", type=float, default=0.1, help="Conformal miscoverage level")
    p.add_argument("--acq-noise-std", type=float, default=3.0,
                   help="Sensor noise std (px-equiv grey levels) for re-acquisition")
    p.add_argument("--max-teeth", type=int, default=3, help="Max scorable teeth per image")
    p.add_argument("--cal-images", type=int, default=200, help="Calibration images cap")
    p.add_argument("--test-images", type=int, default=200, help="Test images cap")
    args = p.parse_args()

    root = Path(args.data)
    if not root.exists():
        print(f"ERROR: DenPAR dataset not found at {root}", file=sys.stderr)
        return 1

    adapter = RealDenparAdapter(root)
    from dcc.landmarks.vitpose_detector import ViTPoseLandmarkDetector
    detector = ViTPoseLandmarkDetector(args.detector_weights, device=args.detector_device)

    cal_records = list(adapter.records("train"))[: args.cal_images]
    test_records = list(adapter.records("test"))[: args.test_images]
    print(f"Calibration images: {len(cal_records)}   Test images: {len(test_records)}")
    print(f"Rendering crestal change delta_px={args.delta_px} into follow-up pixels; "
          f"running ViTPose per timepoint...")

    cal_rows = evaluate_real_image_pairs(
        cal_records, detector, delta_px=args.delta_px,
        acq_noise_std=args.acq_noise_std, max_teeth_per_image=args.max_teeth, seed=1,
    )
    if not cal_rows:
        print("ERROR: no scorable calibration teeth", file=sys.stderr)
        return 1
    conformal = AsymmetricConformalInterval.fit(
        [r["score"] for r in cal_rows], [r["true_change"] for r in cal_rows], alpha=args.alpha,
    )
    print(f"Calibrated conformal: q_lo={conformal.q_lo:.3f} q_hi={conformal.q_hi:.3f} "
          f"(n_cal_rows={len(cal_rows)})")

    test_rows = evaluate_real_image_pairs(
        test_records, detector, delta_px=args.delta_px,
        acq_noise_std=args.acq_noise_std, max_teeth_per_image=args.max_teeth, seed=2,
    )
    summary = summarize_real_image_decisions(test_rows, conformal, tau=args.tau)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "denpar",
        "detector": "vitpose",
        "delta_px": args.delta_px,
        "tau": args.tau,
        "alpha": args.alpha,
        "acq_noise_std": args.acq_noise_std,
        "n_cal_rows": len(cal_rows),
        "n_test_rows": len(test_rows),
        **summary,
        "test_rows": test_rows,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n=== Real-image detector recall (DenPAR test) ===")
    print(f"  delta_px (true change):     {args.delta_px}")
    print(f"  tau (certification thresh): {args.tau}")
    print(f"  n_stable / n_progressed:    {summary['n_stable']} / {summary['n_progressed']}")
    print(f"  true_change_recall:         {summary['true_change_recall']:.3f}")
    print(f"  false_progression_rate:     {summary['false_progression_rate']:.3f}")
    print(f"  stable_certification_rate:  {summary['stable_certification_rate']:.3f}")
    print(f"\n  Metrics: {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
