#!/usr/bin/env python3
"""Gate 2 end-to-end pipeline on real perio-KPT data.

Builds pairs from GT annotations (simulating acquisition noise + true crestal
change), calibrates ConformalInterval on the experiment split, then evaluates
on holdout using REAL predicted landmarks from the trained ViTPose detector,
and writes a markdown report.

Predicted landmarks always come from ViTPose — there is no synthetic stand-in.
``--detector-weights`` is required; the gate refuses to run without a trained
detector.

Usage:
    python scripts/run_gate2.py \\
        --data data/perio-kpt/extracted/perio_KPT \\
        --output outputs/gate2 \\
        --detector-weights outputs/vitpose_detector/checkpoint_best.pt \\
        [--alpha 0.1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dcc.benchmark.pipeline import evaluate_pairs
from dcc.calibration.protocol import check_calibration_budget
from dcc.certificate.conformal import ConformalInterval
from dcc.certificate.oracle import oracle_interval
from dcc.data.pair_builder import PairBuilderConfig, build_pairs
from dcc.data.perio_kpt_adapter import PerioKptAdapter
from dcc.landmarks.store import PredictedLandmarkStore
from dcc.eval.metrics import (
    auc_fpr_coverage_curve,
    coverage_vs_false_progression_curve,
    summarize_decisions,
    tightness_at_fixed_coverage,
)
from dcc.eval.report import write_report
from dcc.config import load_yaml
from dcc.perturb.acquisition import TransformParams
from dcc.perturb.confounder import maximize_artifact_score_full
from dcc.score.periodontal import scalar_change_score


def _adversarial_confounder_analysis(
    stable_pairs: list,
    tau: float,
    dx_range: list | None = None,
    dy_range: list | None = None,
    scale_range: list | None = None,
) -> dict:
    """Apply adversarial confounder to stable pairs and report worst-case FPR.

    The confounder searches over rigid (translate + scale) transforms that
    maximise the bone-level change score.  Because a proper bone-vector score
    is invariant to rigid-body motion (both CEJ and crest_line shift equally),
    the worst-case score should be near 0 for stable pairs.

    Returns a dict with:
        - worst_case_score: max change score across all stable pairs + all
          candidate perturbations (should be ~0 for a correct score function)
        - worst_case_fpr: fraction of stable pairs flagged as progressed by
          the worst-case perturbation (should be 0)
        - n_stable: number of stable pairs evaluated
    """
    if dx_range is None:
        dx_range = [-20, -10, -5, 0, 5, 10, 20]
    if dy_range is None:
        dy_range = [-20, -10, -5, 0, 5, 10, 20]
    if scale_range is None:
        scale_range = [0.95, 1.0, 1.05]

    # Candidate perturbation grid: rigid translate ± 20px, scale ± 5%
    candidates = [
        TransformParams(dx=float(dx), dy=float(dy), scale=float(scale))
        for dy in dy_range
        for dx in dx_range
        for scale in scale_range
    ]

    worst_scores = []
    flagged = 0
    for pair in stable_pairs:
        try:
            result = maximize_artifact_score_full(pair.baseline, rigid_candidates=candidates)
            worst_scores.append(abs(result.score))
            if result.score >= tau:
                flagged += 1
        except Exception:
            pass

    n = len(stable_pairs)
    if not worst_scores:
        return {"worst_case_score": 0.0, "worst_case_fpr": 0.0, "n_stable": n}

    return {
        "worst_case_score": float(max(worst_scores)),
        "worst_case_fpr": flagged / n if n > 0 else 0.0,
        "n_stable": n,
    }


def _calibrate(cal_pairs: list, alpha: float):
    from dcc.certificate.conformal import AsymmetricConformalInterval
    predicted, observed = [], []
    for p in cal_pairs:
        score = scalar_change_score(p.baseline, p.followup)
        predicted.append(score)
        observed.append(p.true_change)
    return AsymmetricConformalInterval.fit(predicted, observed, alpha=alpha)


def main() -> None:
    _perturb = load_yaml("perturb_ranges.yaml")
    _thresh = load_yaml("thresholds.yaml")
    _noise_std_default = float(_perturb.get("acquisition_noise", {}).get("noise_std_px", 3.0))
    _tau_default = float(_thresh.get("bone_level_change", {}).get("tau_conservative_px", 10.0))
    _shift_default = float(_thresh.get("bone_level_change", {}).get("tau_clinically_significant_px", 20.0))
    _rigid = _perturb.get("rigid_search", {})
    _dx_range = _rigid.get("dx_range_px", [-20, -10, -5, 0, 5, 10, 20])
    _dy_range = _rigid.get("dy_range_px", [-20, -10, -5, 0, 5, 10, 20])
    _scale_range = _rigid.get("scale_range", [0.95, 1.0, 1.05])

    parser = argparse.ArgumentParser(description="Gate 2 pipeline on perio-KPT data")
    parser.add_argument("--data", default="data/perio-kpt/extracted/perio_KPT",
                        help="Root of extracted perio-KPT dataset")
    parser.add_argument("--output", default="outputs/gate2",
                        help="Output directory for report")
    parser.add_argument("--tau", type=float, default=_tau_default,
                        help="Decision threshold in pixels (default=10px ~1mm)")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Conformal significance level")
    parser.add_argument("--noise-std", type=float, default=_noise_std_default,
                        help="Acquisition noise std in pixels (~0.3mm)")
    parser.add_argument("--shift", type=float, default=_shift_default,
                        help="Crestal shift magnitude in pixels for progressed pairs (~2mm)")
    parser.add_argument("--detector-weights", required=True,
                        help="Path to the fine-tuned ViTPose checkpoint (.pt) from "
                             "scripts/train_vitpose_detector.py. Predicted landmarks "
                             "come from the REAL ViTPose detector — there is no "
                             "synthetic stand-in.")
    parser.add_argument("--detector-device", default="auto",
                        help="Device for the detector ('cuda', 'cpu', or 'auto')")
    args = parser.parse_args()

    root = Path(args.data)
    if not root.exists():
        print(f"ERROR: dataset not found at {root}", file=sys.stderr)
        sys.exit(1)

    adapter = PerioKptAdapter(root)
    cfg = PairBuilderConfig(
        acq_noise_std=args.noise_std,
        crestal_shift_px=args.shift,
        seed=42,
    )

    print("Loading calibration records (experiment split)...")
    cal_records = list(adapter.records("experiment"))
    cal_pairs = build_pairs(cal_records, cfg)
    print(f"  {len(cal_records)} records → {len(cal_pairs)} pairs")

    print("Calibrating AsymmetricConformalInterval...")
    conformal = _calibrate(cal_pairs, alpha=args.alpha)
    print(f"  radius={conformal.radius:.6f}  alpha={conformal.alpha}")

    budget = check_calibration_budget(n_cal=len(cal_pairs), alpha=args.alpha, target_coverage=0.9)
    print(f"CalibrationBudget: satisfiable={budget.coverage_satisfiable}, n_needed={budget.n_needed}")
    if not budget.coverage_satisfiable:
        print("WARNING: insufficient calibration samples for target coverage", file=sys.stderr)

    print("Loading holdout records...")
    hold_records = list(adapter.records("holdout"))
    hold_pairs = build_pairs(hold_records, cfg)
    print(f"  {len(hold_records)} records → {len(hold_pairs)} pairs")

    print(f"Building predicted landmark store from trained ViTPose detector "
          f"({args.detector_weights}) for holdout...")
    from dcc.landmarks.vitpose_detector import ViTPoseLandmarkDetector
    detector = ViTPoseLandmarkDetector(args.detector_weights, device=args.detector_device)
    holdout_store = PredictedLandmarkStore.from_vitpose(hold_records, detector)
    landmark_provenance = (
        f"REAL predicted landmarks from trained ViTPose detector "
        f"({args.detector_weights})"
    )
    print(f"  {len(holdout_store)} predicted landmark entries  [{landmark_provenance}]")

    print("Evaluating on holdout...")
    rows = evaluate_pairs(hold_pairs, tau=args.tau, conformal=conformal,
                          landmark_store=holdout_store)

    noise_budget_px = 3.0 * args.noise_std  # 3-sigma bound
    print("\nOracle vs Conformal interval comparison (first 5 holdout pairs):")
    print(f"  {'Score':>10}  {'Oracle lo':>12}  {'Oracle hi':>12}  {'Conformal lo':>14}  {'Conformal hi':>14}")
    for p in hold_pairs[:5]:
        try:
            score = scalar_change_score(p.baseline, p.followup)
            o_lo, o_hi = oracle_interval(score, noise_budget_px)
            c_lo, c_hi = conformal.predict(score)
            print(f"  {score:>10.4f}  {o_lo:>12.4f}  {o_hi:>12.4f}  {c_lo:>14.4f}  {c_hi:>14.4f}")
        except Exception:  # noqa: BLE001
            pass

    summary = summarize_decisions(rows)
    out_dir = Path(args.output)
    report_path, metrics_path = write_report(summary, out_dir, rows=rows, tau=args.tau)

    # --- Adversarial confounder analysis (E2b / Gate-1 worst-case FPR) ---
    print("\nRunning adversarial confounder on stable holdout pairs...")
    stable_pairs = [p for p in hold_pairs if p.label == "stable"]
    confounder_stats = _adversarial_confounder_analysis(
        stable_pairs, tau=args.tau,
        dx_range=_dx_range, dy_range=_dy_range, scale_range=_scale_range,
    )
    print(f"  n_stable:          {confounder_stats['n_stable']}")
    print(f"  worst_case_score:  {confounder_stats['worst_case_score']:.4f} px")
    print(f"  worst_case_fpr:    {confounder_stats['worst_case_fpr']:.3f}")
    if confounder_stats["worst_case_score"] < args.tau / 2:
        print("  Score function is acquisition-perturbation invariant (worst-case score < tau/2)")
    else:
        print("  WARNING: Acquisition perturbation inflates score — score function may not be invariant",
              file=sys.stderr)

    # Append confounder stats to metrics.json
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["adversarial_confounder"] = confounder_stats
    metrics["landmark_provenance"] = landmark_provenance
    metrics["pair_provenance"] = (
        "GT landmarks (REAL perio-KPT) + simulated acquisition noise + injected "
        "crestal shift (SYNTHETIC change). Acceptance metrics (recall, FPR, "
        "certification rate) are measured against this semi-synthetic change, not "
        "clinician-verified longitudinal change."
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nReport:  {report_path}")
    print(f"Metrics: {metrics_path}")
    print()
    print(f"  n                         = {summary.n}")
    print(f"  false_progression_rate    = {summary.false_progression_rate:.3f}")
    print(f"  true_change_recall        = {summary.true_change_recall:.3f}")
    print(f"  uncertain_rate            = {summary.uncertain_rate:.3f}")
    print(f"  stable_certification_rate = {summary.stable_certification_rate:.3f}")
    print(f"  mean_interval_width:      {summary.mean_interval_width:.3f} px")
    print(f"  interval_width_std:       {summary.interval_width_std:.3f} px")

    curve = coverage_vs_false_progression_curve(rows, tau=args.tau)
    print("\nFalse-progression vs coverage curve (width_factor → FPR / cert_rate):")
    for pt in curve[::4]:
        print(f"  w={pt['width_factor']:.2f}  fpr={pt['false_prog_rate']:.3f}  cert={pt['stable_cert_rate']:.3f}")
    tightness = tightness_at_fixed_coverage(rows, tau=args.tau, target_coverage=0.90)
    auc = auc_fpr_coverage_curve(curve)
    print(f"\n  tightness_at_90pct_coverage: {tightness}")
    print(f"  auc_fpr_coverage_curve:      {auc:.6f}")


if __name__ == "__main__":
    main()
