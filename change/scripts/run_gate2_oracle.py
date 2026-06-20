#!/usr/bin/env python3
"""Gate-2 conformal certificate with ACCURATE (oracle) landmarks on real DenPAR.

This isolates and validates the *certificate* — the decision system — from the
perception front-end, the standard way to evaluate a decision rule. The
landmarks are the GT (a perfect detector); acquisition uncertainty is injected
via the acquisition-perturbation model and absorbed by the conformal calibration;
a controlled crestal bone-loss change is the true positive signal.

The question answered: given accurate bone-level measurements, does the
acquisition-uncertainty-aware conformal certificate correctly flag a real crestal
change while NOT flagging acquisition-only variation?

    python scripts/run_gate2_oracle.py --data data/denpar/extracted/Dataset \
        --output outputs/gate2_oracle --alpha 0.1 --tau 8 --shift 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dcc.benchmark.pipeline import evaluate_pairs
from dcc.certificate.conformal import AsymmetricConformalInterval
from dcc.data.denpar_adapter import RealDenparAdapter
from dcc.data.pair_builder import PairBuilderConfig, build_pairs
from dcc.eval.metrics import summarize_decisions
from dcc.score.periodontal import scalar_change_score


def calibrate(cal_pairs, alpha):
    predicted = [scalar_change_score(p.baseline, p.followup) for p in cal_pairs]
    observed = [p.true_change for p in cal_pairs]
    return AsymmetricConformalInterval.fit(predicted, observed, alpha=alpha)


def run_one(cal_records, test_records, *, alpha, tau, shift, noise_std, seed=42):
    cfg = PairBuilderConfig(acq_noise_std=noise_std, crestal_shift_px=shift, seed=seed)
    conformal = calibrate(build_pairs(cal_records, cfg), alpha)
    rows = evaluate_pairs(build_pairs(test_records, cfg), tau=tau, conformal=conformal)
    summary = summarize_decisions(rows)
    return conformal, summary


def main() -> int:
    p = argparse.ArgumentParser(description="Gate-2 oracle conformal certificate (DenPAR)")
    p.add_argument("--data", default="data/denpar/extracted/Dataset")
    p.add_argument("--output", default="outputs/gate2_oracle")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=8.0,
                   help="Clinically-significant bone-level change threshold (px)")
    p.add_argument("--shift", type=float, default=20.0,
                   help="Injected crestal change magnitude (px) for the main run")
    p.add_argument("--noise-std", type=float, default=3.0,
                   help="Acquisition perturbation std (px)")
    p.add_argument("--sweep", default="4,8,12,16,20,28,40",
                   help="Crestal-shift magnitudes (px) for the recall-vs-change curve")
    args = p.parse_args()

    root = Path(args.data)
    if not root.exists():
        print(f"ERROR: DenPAR dataset not found at {root}", file=sys.stderr)
        return 1

    adapter = RealDenparAdapter(root)
    cal_records = list(adapter.records("train"))
    test_records = list(adapter.records("test"))
    print(f"Calibration images: {len(cal_records)}   Test images: {len(test_records)}")

    conformal, summary = run_one(
        cal_records, test_records,
        alpha=args.alpha, tau=args.tau, shift=args.shift, noise_std=args.noise_std,
    )
    print(f"\n=== Oracle conformal certificate (DenPAR test, shift={args.shift}px, tau={args.tau}px) ===")
    print(f"  conformal q_lo / q_hi:       {conformal.q_lo:.3f} / {conformal.q_hi:.3f} px")
    print(f"  true_change_recall:          {summary.true_change_recall:.3f}")
    print(f"  false_progression_rate:      {summary.false_progression_rate:.3f}")
    print(f"  stable_certification_rate:   {summary.stable_certification_rate:.3f}")
    print(f"  uncertain_rate:              {summary.uncertain_rate:.3f}")

    sweep = [float(x) for x in args.sweep.split(",")]
    curve = []
    print("\n  Recall / FPR vs injected change magnitude:")
    print(f"  {'shift_px':>9} {'recall':>8} {'FPR':>8} {'stable_cert':>12}")
    for s in sweep:
        _, sm = run_one(cal_records, test_records,
                        alpha=args.alpha, tau=args.tau, shift=s, noise_std=args.noise_std)
        curve.append({"shift_px": s, "recall": sm.true_change_recall,
                      "fpr": sm.false_progression_rate,
                      "stable_cert": sm.stable_certification_rate})
        print(f"  {s:>9.0f} {sm.true_change_recall:>8.3f} {sm.false_progression_rate:>8.3f} "
              f"{sm.stable_certification_rate:>12.3f}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "denpar", "landmarks": "oracle_gt", "alpha": args.alpha,
        "tau": args.tau, "shift": args.shift, "noise_std": args.noise_std,
        "main": {"recall": summary.true_change_recall,
                 "false_progression_rate": summary.false_progression_rate,
                 "stable_certification_rate": summary.stable_certification_rate,
                 "uncertain_rate": summary.uncertain_rate,
                 "q_lo": conformal.q_lo, "q_hi": conformal.q_hi},
        "sweep": curve,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Metrics: {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
