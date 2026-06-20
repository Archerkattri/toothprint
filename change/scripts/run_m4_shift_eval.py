#!/usr/bin/env python3
"""M4 shift evaluation script: standard vs weighted conformal under distribution shift.

Evaluates:
  1. Perturbation-family shift: calibration uses 3px noise, test uses 10px noise
  2. Cross-source shift: calibrated on DenPAR, tested on periapical (or simulated)

Real data is REQUIRED by default: if the real datasets are absent the script
exits non-zero. Synthetic data is only used when you explicitly pass
--allow-synthetic (its numbers are NOT a real-data result).

Usage:
    # real data (default)
    python scripts/run_m4_shift_eval.py \\
        --denpar-root data/denpar/extracted/Dataset \\
        --perio-kpt-root data/perio-kpt/extracted/perio_KPT \\
        [--alpha 0.1] [--cal-noise 3.0] [--test-noise 10.0]

    # explicit synthetic fallback (NOT a real-data result)
    python scripts/run_m4_shift_eval.py --allow-synthetic [--n-synthetic 80]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from dcc.certificate.weighted_conformal import WeightedConformalInterval
from dcc.benchmark.shift_eval import (
    ShiftEvalResult,
    evaluate_under_perturbation_shift,
    evaluate_under_cross_source_shift,
)


# ---------------------------------------------------------------------------
# Synthetic data fallback
# ---------------------------------------------------------------------------

def _make_synthetic_annotation(rng: np.random.Generator, n_teeth: int = 3) -> dict:
    """Build a minimal annotation dict with random landmark coords."""
    teeth = []
    for i in range(n_teeth):
        base_x = rng.uniform(50, 200)
        base_y = rng.uniform(50, 200)
        teeth.append({
            "tooth_id": str(i + 1),
            "cej": [[base_x, base_y], [base_x + 20, base_y + 1]],
            "crest_line": [[base_x + 1, base_y + 30], [base_x + 19, base_y + 31]],
            "apex": [[base_x + 10, base_y + 80]],
        })
    return {"image": f"synthetic_{rng.integers(0, 99999)}.png", "teeth": teeth}


class _SyntheticRecord:
    """Minimal record compatible with build_pairs()."""

    def __init__(self, annotation_dict: dict) -> None:
        self.annotation_dict = annotation_dict
        self.image_id = annotation_dict["image"]


def _build_synthetic_records(n: int, seed: int = 0) -> list[_SyntheticRecord]:
    rng = np.random.default_rng(seed)
    return [_SyntheticRecord(_make_synthetic_annotation(rng)) for _ in range(n)]


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_COL = 24

def _print_row(label: str, std: str, weighted: str) -> None:
    print(f"  {label:<{_COL}} {std:<18} {weighted}")


def _print_result(result: ShiftEvalResult, *, using_real: bool = True) -> None:
    shift = result.shift_type.replace("_", " ").title()
    print(f"\n{'=' * 60}")
    if not using_real:
        print("  *** SOURCE: SYNTHETIC — NOT A REAL-DATA RESULT ***")
    print(f"  Shift type: {shift}")
    print(f"  alpha = {result.alpha}  |  n_test = {result.n_test}")
    print(f"  {'Metric':<{_COL}} {'Standard':>18} {'Weighted':>18}")
    print(f"  {'-' * (_COL + 38)}")
    _print_row(
        "Coverage",
        f"{result.standard_coverage:.4f}",
        f"{result.weighted_coverage:.4f}",
    )
    _print_row(
        "Mean width (px)",
        f"{result.standard_mean_width:.4f}",
        f"{result.weighted_mean_width:.4f}",
    )
    target = 1.0 - result.alpha
    std_ok = result.standard_coverage >= target - 0.05
    w_ok = result.weighted_coverage >= target - 0.05
    print(f"  {'Target coverage':<{_COL}} {target:.4f}")
    print(f"  {'Std meets target':<{_COL}} {'YES' if std_ok else 'NO'}")
    print(f"  {'Wt  meets target':<{_COL}} {'YES' if w_ok else 'NO'}")


# ---------------------------------------------------------------------------
# Real dataset helpers
# ---------------------------------------------------------------------------

def _try_load_denpar_records(denpar_root: Path) -> list | None:
    """Return RealDenparRecord list or None if dataset is absent."""
    if not denpar_root.exists():
        return None
    try:
        from dcc.data.denpar_adapter import load_real_denpar
        records = load_real_denpar(denpar_root)
        if not records:
            return None
        return records
    except Exception as exc:
        print(f"  [warn] Could not load DenPAR: {exc}", file=sys.stderr)
        return None


def _try_load_perio_kpt_records(perio_root: Path) -> list | None:
    """Return perio-KPT baseline records (real second source) or None if absent."""
    if not perio_root.exists():
        return None
    try:
        from dcc.data.perio_kpt_adapter import PerioKptAdapter
        records = list(PerioKptAdapter(perio_root).records("baseline"))
        if not records:
            return None
        return records
    except Exception as exc:
        print(f"  [warn] Could not load perio-KPT: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="M4 shift evaluation: standard vs weighted conformal")
    parser.add_argument("--denpar-root", default="data/denpar/extracted/Dataset",
                        help="Path to DenPAR Dataset/ directory")
    parser.add_argument("--perio-kpt-root", default="data/perio-kpt/extracted/perio_KPT",
                        help="Path to perio-KPT root (real second source for cross-source shift)")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="Permit the synthetic-data fallback when real datasets are "
                             "absent. WITHOUT this flag, real data is REQUIRED and the "
                             "script exits non-zero if it is missing (synthetic results "
                             "are not real-data results and must be opted into).")
    parser.add_argument("--require-real", action="store_true",
                        help="Deprecated: real data is now required by default. Kept as a "
                             "no-op alias (it is the inverse of --allow-synthetic).")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Miscoverage level (default: 0.1)")
    parser.add_argument("--cal-noise", type=float, default=3.0,
                        help="Calibration acquisition noise std in pixels")
    parser.add_argument("--test-noise", type=float, default=10.0,
                        help="Test acquisition noise std in pixels (shifted)")
    parser.add_argument("--n-synthetic", type=int, default=80,
                        help="Number of synthetic records when real data absent")
    args = parser.parse_args()

    print("=" * 60)
    print("  M4 — Weighted Conformal Under Covariate Shift")
    print("=" * 60)

    from dcc.data.pair_builder import PairBuilderConfig, build_pairs

    # Real data is the DEFAULT. Synthetic is only used when explicitly opted in
    # via --allow-synthetic (--require-real is kept as a deprecated alias for the
    # default-strict behavior).
    allow_synthetic = args.allow_synthetic and not args.require_real

    # --- Try real data first ---
    denpar_root = Path(args.denpar_root)
    denpar_records = _try_load_denpar_records(denpar_root)
    using_real = denpar_records is not None and len(denpar_records) >= 4

    if not using_real and not allow_synthetic:
        print(
            f"ERROR: real DenPAR not found at {denpar_root} (or <4 records).\n"
            "Real data is required by default. Extract the dataset (see docs/DATA.md)\n"
            "or pass --allow-synthetic to EXPLICITLY run on the synthetic fallback\n"
            "(whose numbers are NOT a real-data result).",
            file=sys.stderr,
        )
        sys.exit(1)

    if using_real:
        print(f"\n  Loaded {len(denpar_records)} DenPAR records from {denpar_root}")
        # Split: first 60% calibration, remaining test
        n_cal = max(2, int(len(denpar_records) * 0.6))
        cal_records = denpar_records[:n_cal]
        test_records = denpar_records[n_cal:]
        source_label = "denpar"
    else:
        n = args.n_synthetic
        print(f"\n  DenPAR not found at {denpar_root} — using {n} synthetic records")
        all_records = _build_synthetic_records(n, seed=42)
        n_cal = max(2, int(n * 0.6))
        cal_records = all_records[:n_cal]
        test_records = all_records[n_cal:]
        source_label = "synthetic"

    # --- Experiment 1: Perturbation-family shift ---
    print("\n[1] Perturbation-family shift")
    print(f"    Calibration noise std = {args.cal_noise} px  |  Test noise std = {args.test_noise} px")

    cal_cfg_low = PairBuilderConfig(acq_noise_std=args.cal_noise, crestal_shift_px=20.0, seed=0)
    test_cfg_high = PairBuilderConfig(acq_noise_std=args.test_noise, crestal_shift_px=20.0, seed=1)

    cal_pairs_low = build_pairs(cal_records, cal_cfg_low)
    test_pairs_high = build_pairs(test_records, test_cfg_high)

    print(f"    cal_pairs={len(cal_pairs_low)}  test_pairs={len(test_pairs_high)}")

    result_perturb = evaluate_under_perturbation_shift(
        cal_pairs=cal_pairs_low,
        test_pairs=test_pairs_high,
        alpha=args.alpha,
        cal_noise_std=args.cal_noise,
        test_noise_std=args.test_noise,
    )
    _print_result(result_perturb, using_real=using_real)

    # --- Experiment 2: Cross-source shift ---
    print("\n[2] Cross-source shift")

    # The real second source is perio-KPT (a distinct dataset from the unused
    # periapical-lesions set); label it "perio_kpt" honestly so the importance
    # weights are not conflated with periapical lesions.
    test_source = "perio_kpt"
    test_cfg_perio = PairBuilderConfig(acq_noise_std=args.test_noise, crestal_shift_px=20.0, seed=2)

    # A genuine importance-weighting comparison requires calibration to contain
    # BOTH the test source (perio_kpt, weight 1.0) AND a different source
    # (denpar/synthetic, weight downweight_ratio).  We load perio-KPT as the real
    # second source; only then are the weights non-uniform and the
    # weighted-vs-standard comparison meaningful.
    perio_root = Path(args.perio_kpt_root)
    perio_records = _try_load_perio_kpt_records(perio_root)

    cross_using_real = (
        using_real and perio_records is not None and len(perio_records) >= 4
    )
    if perio_records is not None and len(perio_records) >= 4:
        # Split real perio-KPT records into disjoint cal/test halves (no leakage).
        n_perio_cal = len(perio_records) // 2
        perio_cal_records = perio_records[:n_perio_cal]
        perio_test_records = perio_records[n_perio_cal:]

        perio_cal_cfg = PairBuilderConfig(acq_noise_std=args.cal_noise, crestal_shift_px=20.0, seed=3)
        perio_cal_pairs = build_pairs(perio_cal_records, perio_cal_cfg)
        test_pairs_perio = build_pairs(perio_test_records, test_cfg_perio)

        # Mixed-source calibration: denpar/synthetic (downweighted) + real perio-KPT.
        cal_pairs_cross = list(cal_pairs_low) + list(perio_cal_pairs)
        cal_sources = ([source_label] * len(cal_pairs_low)) + (["perio_kpt"] * len(perio_cal_pairs))
        print(f"    Calibration sources: {len(cal_pairs_low)}x '{source_label}' (downweighted) "
              f"+ {len(perio_cal_pairs)}x real 'perio_kpt'")
        print(f"    Test source='{test_source}' (real perio-KPT)  test_pairs={len(test_pairs_perio)}")
    else:
        # No real second source: the comparison is DEGENERATE (uniform weights).
        if not allow_synthetic:
            print(
                f"ERROR: real perio-KPT not found at {perio_root} (or <4 records).\n"
                "Real data is required by default for the cross-source experiment.\n"
                "Extract perio-KPT (see docs/DATA.md) or pass --allow-synthetic to\n"
                "EXPLICITLY run the DEGENERATE/SIMULATED fallback.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"    WARNING: no real perio_kpt source at {perio_root}; cross-source is "
              f"DEGENERATE/SIMULATED — all calibration weights collapse to a single value, "
              f"so weighted conformal == standard. Do not cite this as a real reweighting result.",
              file=sys.stderr)
        cal_pairs_cross = cal_pairs_low
        cal_sources = [source_label] * len(cal_pairs_low)
        test_pairs_perio = build_pairs(test_records, test_cfg_perio)
        print(f"    Calibration source='{source_label}'  Test source='{test_source}'")
        print(f"    cal_pairs={len(cal_pairs_cross)}  test_pairs={len(test_pairs_perio)}")

    result_cross = evaluate_under_cross_source_shift(
        cal_pairs=cal_pairs_cross,
        cal_sources=cal_sources,
        test_pairs=test_pairs_perio,
        test_source=test_source,
        alpha=args.alpha,
        downweight_ratio=0.1,
    )
    _print_result(result_cross, using_real=cross_using_real)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
