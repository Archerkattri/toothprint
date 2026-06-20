#!/usr/bin/env python3
"""DentalMapCert Gate 1 demo pipeline with fully synthetic capture sessions.

No real dataset required. Generates N synthetic subjects, each with two
timepoints (t0 / t1) and a random subset of the five standard protocol views.
Calibrates ErrorCalibrator on training subjects and issues certificates for
test subjects.

Configurable generation parameters
------------------------------------
The sampling probabilities used during synthetic subject generation are
deliberately exposed as named constants below (see ``_P_OPTIONAL_VIEW``,
``_P_DROP_REQUIRED_T1``, ``_P_QUALITY_ISSUE``) rather than being embedded as
unnamed literals.  Adjust these to model different clinical populations:

- ``_P_OPTIONAL_VIEW``    — probability that an optional view is captured in
  either timepoint (default 0.60, i.e. operators include occlusal views ~60%
  of the time).
- ``_P_DROP_REQUIRED_T1`` — probability that one required view is missing at t1
  (default 0.20, modelling a 1-in-5 incomplete follow-up session).
- ``_P_QUALITY_ISSUE``    — probability that any given view has a quality
  problem (blur or glare; default 0.25).

Calibration residual range
--------------------------
Synthetic reconstruction residuals are drawn uniformly from
``[_RESIDUAL_MIN_MM, _RESIDUAL_MAX_MM]`` (default 0.05–0.45 mm), which spans
the typical visible-surface reconstruction error range for intraoral photos
reconstructed with learned stereo methods.

Usage:
    python scripts/run_dmc_gate1.py --output outputs/dmc_gate1 [--subjects 40]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dentalmapcert.calibration import ErrorCalibrator
from dentalmapcert.capture_protocol import STANDARD_PROTOCOL, coverage_per_region
from dentalmapcert.certificate import CertificateInput, decide_surface_change
from dentalmapcert.report import write_outputs


# -----------------------------------------------------------------------
# Tiny deterministic LCG so we don't import random (keeps it reproducible)
# -----------------------------------------------------------------------

def _lcg(state: int) -> tuple[int, float]:
    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
    return state, state / 0xFFFFFFFF


def _uniform(state: int, lo: float, hi: float) -> tuple[int, float]:
    s, u = _lcg(state)
    return s, lo + u * (hi - lo)


def _choice(state: int, items: list) -> tuple[int, object]:
    s, u = _lcg(state)
    return s, items[int(u * len(items))]


# -----------------------------------------------------------------------
# Synthetic subject generation
# -----------------------------------------------------------------------

ALL_VIEWS = [v.view_name for v in STANDARD_PROTOCOL]
REQUIRED_VIEWS = [v.view_name for v in STANDARD_PROTOCOL if v.required]
OPTIONAL_VIEWS = [v.view_name for v in STANDARD_PROTOCOL if not v.required]

# ---------------------------------------------------------------------------
# Configurable generation parameters (see module docstring for guidance)
# ---------------------------------------------------------------------------

# Probability that each optional view is present in a capture session.
_P_OPTIONAL_VIEW: float = 0.60

# Probability that one required view is dropped at t1 (incomplete follow-up).
_P_DROP_REQUIRED_T1: float = 0.20

# Probability that any given view has a quality issue (blur or glare).
_P_QUALITY_ISSUE: float = 0.25

# Reconstruction residual range (mm) for calibration samples.
_RESIDUAL_MIN_MM: float = 0.05
_RESIDUAL_MAX_MM: float = 0.45


def _make_subject(subject_idx: int, seed: int) -> dict:
    """Return a dict with t0/t1 view lists and quality tags.

    The sampling probabilities (``_P_OPTIONAL_VIEW``, ``_P_DROP_REQUIRED_T1``,
    ``_P_QUALITY_ISSUE``) and residual range (``_RESIDUAL_MIN_MM``,
    ``_RESIDUAL_MAX_MM``) are module-level constants — adjust them to model
    different clinical populations without touching this function.
    """
    state = seed ^ (subject_idx * 0x9E3779B9)

    # t0: always has required views; _P_OPTIONAL_VIEW chance of each optional view.
    t0_views = list(REQUIRED_VIEWS)
    for opt in OPTIONAL_VIEWS:
        state, u = _lcg(state)
        if u < _P_OPTIONAL_VIEW:
            t0_views.append(opt)

    # t1: same protocol, but _P_DROP_REQUIRED_T1 chance of one required view absent.
    t1_views = list(REQUIRED_VIEWS)
    state, u = _lcg(state)
    if u < _P_DROP_REQUIRED_T1 and t1_views:
        state, drop_idx = _uniform(state, 0, len(t1_views) - 0.001)
        t1_views.pop(int(drop_idx))
    for opt in OPTIONAL_VIEWS:
        state, u = _lcg(state)
        if u < _P_OPTIONAL_VIEW:
            t1_views.append(opt)

    # Assign quality tags: each view has _P_QUALITY_ISSUE chance of blur or glare.
    def _tags(views: list[str]) -> dict[str, list[str]]:
        nonlocal state
        result: dict[str, list[str]] = {}
        for v in views:
            state, u = _lcg(state)
            if u < _P_QUALITY_ISSUE:
                state, issue = _choice(state, ["glare", "blur"])
                result[v] = [issue]
        return result

    t0_tags = _tags(t0_views)
    t1_tags = _tags(t1_views)

    # Synthetic reconstruction residual (mm) — uniform over the configured range.
    state, residual_mm = _uniform(state, _RESIDUAL_MIN_MM, _RESIDUAL_MAX_MM)

    return {
        "subject_id": f"S{subject_idx:03d}",
        "t0_views": t0_views,
        "t1_views": t1_views,
        "t0_tags": t0_tags,
        "t1_tags": t1_tags,
        "residual_mm": round(residual_mm, 4),
    }


# -----------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DentalMapCert Gate 1 synthetic demo")
    parser.add_argument("--output", default="outputs/dmc_gate1", help="Output directory")
    parser.add_argument("--subjects", type=int, default=40, help="Total synthetic subjects")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--alpha", type=float, default=0.1, help="Conformal significance level")
    parser.add_argument("--coverage-threshold", type=float, default=0.15,
                        help="Minimum coverage fraction to issue a certificate "
                             "(use 0.15 for synthetic heuristic; 0.75 for real reconstruction)")
    parser.add_argument("--stable-threshold-mm", type=float, default=0.50)
    parser.add_argument("--change-threshold-mm", type=float, default=0.90)
    args = parser.parse_args()

    n = args.subjects
    n_cal = max(4, n * 2 // 3)
    n_test = n - n_cal

    print(f"Generating {n} subjects ({n_cal} calibration, {n_test} test)...")
    subjects = [_make_subject(i, args.seed) for i in range(n)]

    # Calibrate on first n_cal subjects
    cal_residuals = [s["residual_mm"] for s in subjects[:n_cal]]
    calibrator = ErrorCalibrator.fit(cal_residuals, alpha=args.alpha)
    print(f"ErrorCalibrator: radius_mm={calibrator.radius_mm:.4f}  alpha={calibrator.alpha}")

    # Build certificates for test subjects
    REGIONS = ["anterior_crown", "buccal_crown", "visible_gingival_margin"]
    certificates = []
    for subject in subjects[n_cal:]:
        t0_cov = coverage_per_region(subject["t0_views"], subject["t0_tags"])
        t1_cov = coverage_per_region(subject["t1_views"], subject["t1_tags"])

        for region in REGIONS:
            cov_t0 = t0_cov.get(region, 0.0)
            cov_t1 = t1_cov.get(region, 0.0)

            # Synthetic surface change estimate (mm) — LCG placeholder intentional here.
            # This script is the fully-synthetic demo; it does not use real images.
            # For real image data, run_dmc_gate1_real.py wires surface_error_mm() from
            # the actual reconstruct_point_cloud → surface_error_mm pipeline.
            # ~60% stable subjects (0-0.1mm), ~40% changing (0.8-1.5mm)
            state = args.seed ^ hash(subject["subject_id"] + region)
            state, u0 = _lcg(state)
            state, u1 = _lcg(state)
            if u0 < 0.6:
                estimate_mm = u1 * 0.1   # stable: 0-0.1mm
            else:
                estimate_mm = 0.8 + u1 * 0.7  # progressed: 0.8-1.5mm

            err_t0 = calibrator.interval(0.0)
            err_t1 = calibrator.interval(0.0)
            # delta interval = estimate ± radius (one radius for the change measurement)
            delta_lo = max(0.0, estimate_mm - calibrator.radius_mm)
            delta_hi = estimate_mm + calibrator.radius_mm

            inp = CertificateInput(
                surface_region_id=f"{subject['subject_id']}_{region}",
                capture_id_t0=f"{subject['subject_id']}_t0",
                capture_id_t1=f"{subject['subject_id']}_t1",
                coverage_score_t0=cov_t0,
                coverage_score_t1=cov_t1,
                error_interval_mm_t0=err_t0,
                error_interval_mm_t1=err_t1,
                delta_interval_mm=(round(delta_lo, 6), round(delta_hi, 6)),
                region_type=region if region in (
                    "anterior_crown", "buccal_crown", "visible_gingival_margin"
                ) else "unknown_visible_dental_surface",
            )
            cert = decide_surface_change(
                inp,
                coverage_threshold=args.coverage_threshold,
                stable_threshold_mm=args.stable_threshold_mm,
                change_threshold_mm=args.change_threshold_mm,
            )
            certificates.append(cert)

    out_dir = Path(args.output)
    report_path, jsonl_path = write_outputs(certificates, out_dir)

    # Summary
    counts: dict[str, int] = {}
    for cert in certificates:
        counts[cert.label] = counts.get(cert.label, 0) + 1

    print(f"\nReport:  {report_path}")
    print(f"Records: {jsonl_path}")
    print(f"\nTotal certificates: {len(certificates)}")
    for label, n_label in sorted(counts.items()):
        print(f"  {label}: {n_label} ({100*n_label/len(certificates):.0f}%)")


if __name__ == "__main__":
    main()
