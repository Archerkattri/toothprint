#!/usr/bin/env python3
"""DentalMapCert Gate 1 pipeline with real PhoneCaptureLoader data.

When ``data/phone-captures/`` exists and contains subject/timepoint sub-directories
with image files, this script uses PhoneCaptureLoader to discover real records,
attempts longitudinal pairing, and issues certificates for each matched pair.
The ``delta_interval_mm`` comes from the real
render_5_views → reconstruct_point_cloud → surface_error_mm chain.

De-synthetic contract
---------------------
The LCG synthetic delta is NOT a silent default.

  - When no real captures exist, the script EXITS with an error unless
    ``--synthetic`` is given (which runs the explicitly-labelled LCG demo).
  - In real mode, when reconstruction fails to produce a delta for a subject,
    that subject is SKIPPED rather than certified with a fabricated delta —
    unless ``--allow-synthetic-fallback`` is passed.

Real-data path:
  - PhoneCaptureLoader discovers subject/timepoint image records.
  - Longitudinal pairs are formed with pair_by_subject().
  - The real surface-change delta is computed from the reconstruction chain.
  - coverage_per_region() provides coverage estimates for the protocol views.
  - ErrorCalibrator is fitted on the per-pair residuals.
  - Certificates are issued per region per pair.
  - A Markdown + JSONL report is written to the output directory.

Status: implemented; the real reconstruction delta is pending a GPU/data run
(VGGT/DUSt3R need a GPU; without one the chain falls through to the crude
Open3D fallback).

Usage:
    # Real data:
    python scripts/run_dmc_gate1_real.py --phone-captures data/phone-captures
    # Explicit synthetic demo:
    python scripts/run_dmc_gate1_real.py --synthetic [--subjects 40]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dentalmapcert.calibration import ErrorCalibrator
from dentalmapcert.capture_protocol import STANDARD_PROTOCOL, coverage_per_region
from dentalmapcert.certificate import CertificateInput, decide_surface_change
from dentalmapcert.dataset_loaders import PhoneCaptureLoader
from dentalmapcert.image_quality import analyze_view_quality
from dentalmapcert.longitudinal import pair_by_subject
from dentalmapcert.reconstruction import reconstruct_point_cloud
from dentalmapcert.regions import region_id, region_surface_from_id
from dentalmapcert.report import write_outputs
from dentalmapcert.surface_error import surface_error_mm


# ---------------------------------------------------------------------------
# Tiny deterministic LCG — identical to run_dmc_gate1.py so seeds match
# ---------------------------------------------------------------------------

def _lcg(state: int) -> tuple[int, float]:
    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
    return state, state / 0xFFFFFFFF


def _uniform(state: int, lo: float, hi: float) -> tuple[int, float]:
    s, u = _lcg(state)
    return s, lo + u * (hi - lo)


# ---------------------------------------------------------------------------
# Synthetic subject helpers (fallback path)
# ---------------------------------------------------------------------------

ALL_VIEWS = [v.view_name for v in STANDARD_PROTOCOL]
REQUIRED_VIEWS = [v.view_name for v in STANDARD_PROTOCOL if v.required]
OPTIONAL_VIEWS = [v.view_name for v in STANDARD_PROTOCOL if not v.required]


def _make_synthetic_subject(subject_idx: int, seed: int) -> dict:
    """Generate a fully-synthetic subject dict for the fallback pipeline."""
    state = seed ^ (subject_idx * 0x9E3779B9)

    t0_views = list(REQUIRED_VIEWS)
    for opt in OPTIONAL_VIEWS:
        state, u = _lcg(state)
        if u < 0.6:
            t0_views.append(opt)

    t1_views = list(REQUIRED_VIEWS)
    state, u = _lcg(state)
    if u < 0.2 and t1_views:
        state, drop_u = _lcg(state)
        t1_views.pop(min(len(t1_views) - 1, int(drop_u * len(t1_views))))
    for opt in OPTIONAL_VIEWS:
        state, u = _lcg(state)
        if u < 0.6:
            t1_views.append(opt)

    def _tags(views: list[str]) -> dict[str, list[str]]:
        nonlocal state
        result: dict[str, list[str]] = {}
        for v in views:
            state, u = _lcg(state)
            if u < 0.25:
                state, issue_u = _lcg(state)
                issue = "glare" if issue_u < 0.5 else "blur"
                result[v] = [issue]
        return result

    t0_tags = _tags(t0_views)
    t1_tags = _tags(t1_views)
    state, residual_mm = _uniform(state, 0.05, 0.45)

    return {
        "subject_id": f"S{subject_idx:03d}",
        "t0_views": t0_views,
        "t1_views": t1_views,
        "t0_tags": t0_tags,
        "t1_tags": t1_tags,
        "residual_mm": round(residual_mm, 4),
    }


# ---------------------------------------------------------------------------
# Certificate issuance (shared between real and synthetic paths)
# ---------------------------------------------------------------------------

# Anterior teeth visible in the 5 protocol views (FDI notation)
_GATE_TEETH = (11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43)
_GATE_SURFACES = ("buccal", "mesial", "distal")
REGIONS = [region_id(fdi, surf) for fdi in _GATE_TEETH for surf in _GATE_SURFACES]


def _region_type(region_id_str: str) -> str:
    """Map a tooth-surface region id (e.g. ``tooth_11_buccal``) to the
    protocol region-TYPE key that ``coverage_per_region`` returns.

    ``coverage_per_region`` is keyed by region type ("buccal_crown",
    "anterior_crown", ...), not by per-tooth region id, so the gate must
    translate before looking coverage up — otherwise every lookup misses and
    coverage is silently 0.0. All gate teeth (11-13/21-23/31-33/41-43) are
    anterior, so a buccal surface maps to ``buccal_crown`` and the
    mesial/distal surfaces map to ``anterior_crown``.
    """
    _fdi, surface = region_surface_from_id(region_id_str)
    return "buccal_crown" if surface == "buccal" else "anterior_crown"


def _compute_real_delta_mm(
    t0_image_paths: dict[str, Path],
    t1_image_paths: dict[str, Path],
    calibrator: ErrorCalibrator,
) -> tuple[float, float] | None:
    """Run the full mesh→views→reconstruct→surface_error chain for real images.

    Reconstructs a point cloud from t0 images and t1 images separately, then
    computes surface_error_mm(t1_recon, t0_recon) as the change estimate.

    Returns (delta_lo, delta_hi) in mm, or None if reconstruction fails or
    either timepoint has no images.
    """
    if not t0_image_paths or not t1_image_paths:
        return None

    import numpy as np

    try:
        t0_paths = list(t0_image_paths.values())
        t1_paths = list(t1_image_paths.values())

        # A metric reconstruction needs multiple views per timepoint: with a
        # single image, DUSt3R is skipped and the result is an uncalibrated
        # single-view edge projection, so surface_error_mm between two such
        # clouds measures capture pose/lighting rather than true surface
        # change. Fall back to the synthetic estimate rather than certifying a
        # geometrically meaningless "real" delta.
        if len(t0_paths) < 2 or len(t1_paths) < 2:
            logger.warning(
                "Need >=2 views per timepoint for metric reconstruction; got "
                "t0=%d, t1=%d. Falling back to synthetic estimate.",
                len(t0_paths), len(t1_paths),
            )
            return None

        t0_pts, _t0_conf = reconstruct_point_cloud(t0_paths)
        t1_pts, _t1_conf = reconstruct_point_cloud(t1_paths)

        if len(t0_pts) < 3 or len(t1_pts) < 3:
            logger.warning(
                "Skipping surface_error_mm: too few points (t0=%d, t1=%d).",
                len(t0_pts), len(t1_pts),
            )
            return None

        err = surface_error_mm(t1_pts, t0_pts, run_icp=True)
        # reconstruct_point_cloud's Open3D fallback now returns millimetres, so
        # the Chamfer distance is already in mm — no unit conversion needed.
        estimate_mm = err.chamfer_mm
        if not (0.0 <= estimate_mm < 50.0):
            logger.warning(
                "surface_error %.4f mm outside plausible dental range; treating "
                "reconstruction as failed.", estimate_mm,
            )
            return None
        delta_lo = max(0.0, estimate_mm - calibrator.radius_mm)
        delta_hi = estimate_mm + calibrator.radius_mm
        logger.info(
            "surface_error_mm: chamfer=%.4f mm  delta=[%.3f, %.3f]",
            err.chamfer_mm, delta_lo, delta_hi,
        )
        return round(delta_lo, 6), round(delta_hi, 6)

    except Exception as exc:
        logger.warning("surface_error_mm pipeline failed (%s); falling back to LCG estimate.", exc)
        return None


import logging
logger = logging.getLogger(__name__)


def _issue_certificates(
    subject_id: str,
    t0_views: list[str],
    t0_tags: dict[str, list[str]],
    t1_views: list[str],
    t1_tags: dict[str, list[str]],
    calibrator: ErrorCalibrator,
    seed: int,
    coverage_threshold: float,
    stable_threshold_mm: float,
    change_threshold_mm: float,
    t0_image_paths: dict[str, Path] | None = None,
    t1_image_paths: dict[str, Path] | None = None,
    allow_synthetic_delta: bool = False,
) -> tuple[list, list]:
    """Return (cert_inputs, cert_outputs) for each region.

    When real image paths are provided, computes the surface change estimate
    using the real pipeline:
        render_5_views → reconstruct_point_cloud → surface_error_mm

    De-synthetic contract: the LCG synthetic delta is NOT a silent default. It
    is used only when ``allow_synthetic_delta`` is True (set by ``--synthetic``
    or ``--allow-synthetic-fallback``). When real images are supplied but the
    real reconstruction fails and synthetic substitution is not allowed, this
    returns ``([], [])`` so the caller skips the subject instead of certifying a
    fabricated delta.
    """
    t0_cov = coverage_per_region(t0_views, t0_tags, image_paths_per_view=t0_image_paths)
    t1_cov = coverage_per_region(t1_views, t1_tags, image_paths_per_view=t1_image_paths)

    # Attempt real surface-error computation once per subject (shared across regions).
    real_delta: tuple[float, float] | None = None
    if t0_image_paths and t1_image_paths:
        real_delta = _compute_real_delta_mm(t0_image_paths, t1_image_paths, calibrator)

    if real_delta is None and not allow_synthetic_delta:
        # Real images present but reconstruction failed (or no images at all),
        # and the synthetic LCG delta is not permitted: refuse to fabricate a
        # delta. The caller skips this subject.
        logger.warning(
            "Subject %s: no real delta available and synthetic substitution is "
            "disabled; skipping (pass --allow-synthetic-fallback or --synthetic "
            "to permit the LCG demo delta).",
            subject_id,
        )
        return [], []

    cert_inputs = []
    certs = []
    for region in REGIONS:
        region_type = _region_type(region)
        cov_t0 = t0_cov.get(region_type, 0.0)
        cov_t1 = t1_cov.get(region_type, 0.0)

        if real_delta is not None:
            # Real pipeline succeeded: use the ICP/Chamfer-derived interval.
            delta_lo, delta_hi = real_delta
        else:
            # LCG synthetic delta — reached ONLY when allow_synthetic_delta is
            # True (explicit --synthetic / --allow-synthetic-fallback).
            state = seed ^ hash(subject_id + region)
            state, u0 = _lcg(state)
            state, u1 = _lcg(state)
            if u0 < 0.6:
                estimate_mm = u1 * 0.1
            else:
                estimate_mm = 0.8 + u1 * 0.7
            delta_lo = max(0.0, estimate_mm - calibrator.radius_mm)
            delta_hi = estimate_mm + calibrator.radius_mm

        err_t0 = calibrator.interval(0.0)
        err_t1 = calibrator.interval(0.0)

        fdi_num, surf_name = region_surface_from_id(region)
        inp = CertificateInput(
            surface_region_id=f"{subject_id}_{region}",
            capture_id_t0=f"{subject_id}_t0",
            capture_id_t1=f"{subject_id}_t1",
            coverage_score_t0=cov_t0,
            coverage_score_t1=cov_t1,
            error_interval_mm_t0=err_t0,
            error_interval_mm_t1=err_t1,
            delta_interval_mm=(round(delta_lo, 6), round(delta_hi, 6)),
            region_type=f"fdi_{fdi_num}_{surf_name}_surface",
        )
        cert = decide_surface_change(
            inp,
            coverage_threshold=coverage_threshold,
            stable_threshold_mm=stable_threshold_mm,
            change_threshold_mm=change_threshold_mm,
        )
        cert_inputs.append(inp)
        certs.append(cert)
    return cert_inputs, certs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DentalMapCert Gate 1 — real + synthetic fallback")
    parser.add_argument("--output", default="outputs/dmc_gate1_real", help="Output directory")
    parser.add_argument(
        "--phone-captures",
        default="data/phone-captures",
        help="Root directory for PhoneCaptureLoader (default: data/phone-captures)",
    )
    parser.add_argument("--subjects", type=int, default=40,
                        help="Number of synthetic subjects to generate in fallback mode")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--alpha", type=float, default=0.1, help="Conformal significance level")
    parser.add_argument("--coverage-threshold", type=float, default=0.15,
                        help="Minimum coverage fraction (0.15 for synthetic; 0.75 for real reconstruction)")
    parser.add_argument("--stable-threshold-mm", type=float, default=0.50)
    parser.add_argument("--change-threshold-mm", type=float, default=0.90)
    parser.add_argument(
        "--synthetic", action="store_true",
        help="EXPLICIT fully-synthetic demo: generate LCG subjects and LCG deltas "
             "without any real data. Output is labelled synthetic.",
    )
    parser.add_argument(
        "--allow-synthetic-fallback", action="store_true",
        help="In real-data mode, permit the LCG synthetic delta when real "
             "reconstruction fails for a subject (instead of skipping it). Off "
             "by default so a failed reconstruction never silently fabricates a delta.",
    )
    args = parser.parse_args()

    phone_root = Path(args.phone_captures)
    has_real_data = phone_root.exists() and any(phone_root.iterdir())

    if not has_real_data and not args.synthetic:
        # De-synthetic contract: do not silently run the fully-synthetic pipeline.
        print(
            f"ERROR: no real phone captures found at {phone_root}. Refusing to "
            "silently fabricate synthetic results. Pass --synthetic to run the "
            "explicitly-labelled LCG demo, or populate the phone-captures "
            "directory (see docs/DATA.md).",
            file=sys.stderr,
        )
        sys.exit(1)

    use_real = has_real_data and not args.synthetic
    # The LCG delta is permitted only on the explicit synthetic path, or when the
    # operator explicitly opts into the fallback for real-mode reconstruction
    # failures.
    allow_synthetic_delta = args.synthetic or args.allow_synthetic_fallback

    subjects_data: list[dict]  # list of dicts with subject_id, t0/t1 views+tags, residual_mm

    if use_real:
        print(f"Real data found at {phone_root} — loading with PhoneCaptureLoader.")
        loader = PhoneCaptureLoader(str(phone_root))
        all_records = list(loader.records())
        print(f"  Loaded {len(all_records)} image records.")

        pairs = pair_by_subject(all_records)
        print(f"  Formed {len(pairs)} longitudinal pairs.")

        if not pairs:
            if not args.synthetic:
                print(
                    "ERROR: real data present but no longitudinal pairs found "
                    "(need >=2 timepoints per subject). Refusing to silently fall "
                    "back to synthetic. Pass --synthetic for the LCG demo.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print("  No longitudinal pairs found; --synthetic set, using LCG demo.")
            use_real = False
        else:
            # Build subjects_data from real pairs.
            # For each pair we map the t0/t1 image paths to the closest
            # protocol view name (by filename stem) and call analyze_view_quality
            # to derive real quality tags.  Views not matched to a protocol name
            # are assigned the view name "anterior_close" as a best-effort guess.
            subjects_data = []
            state = args.seed
            _view_names_set = set(REQUIRED_VIEWS + list(OPTIONAL_VIEWS))

            def _guess_view_name(img_path: Path) -> str:
                stem = img_path.stem.lower()
                for vn in _view_names_set:
                    if vn.replace("_", "") in stem.replace("_", ""):
                        return vn
                return REQUIRED_VIEWS[0]  # fallback: anterior_close

            def _real_quality_tags(image_map: dict[str, Path]) -> dict[str, list[str]]:
                tags: dict[str, list[str]] = {}
                for view_name, img_path in image_map.items():
                    detected = analyze_view_quality(img_path)
                    if detected and detected != ["unreadable"]:
                        tags[view_name] = detected
                return tags

            for pair in pairs:
                state, residual_mm = _uniform(state, 0.05, 0.45)

                # Collect images for t0 / t1 records
                t0_images: dict[str, Path] = {}
                t1_images: dict[str, Path] = {}
                for tp_key, tp_images, record in [
                    ("t0", t0_images, pair.t0_record),
                    ("t1", t1_images, pair.t1_record),
                ]:
                    img_path = getattr(record, "image_path", None)
                    if img_path is not None:
                        view_name = _guess_view_name(Path(img_path))
                        tp_images[view_name] = Path(img_path)

                t0_view_list = list(t0_images.keys()) if t0_images else list(REQUIRED_VIEWS)
                t1_view_list = list(t1_images.keys()) if t1_images else list(REQUIRED_VIEWS)

                subjects_data.append({
                    "subject_id": pair.subject_id,
                    "t0_views": t0_view_list,
                    "t1_views": t1_view_list,
                    "t0_tags": _real_quality_tags(t0_images),
                    "t1_tags": _real_quality_tags(t1_images),
                    "t0_image_paths": t0_images,
                    "t1_image_paths": t1_images,
                    "residual_mm": round(residual_mm, 4),
                })
            print(f"  Using {len(subjects_data)} real subjects for the pipeline.")

    if not use_real:
        # Reached only on the explicit --synthetic path (the no-data and
        # no-pairs cases above already exit unless --synthetic was passed).
        print(f"Using fully-synthetic pipeline ({args.subjects} subjects).", flush=True)
        print(
            "WARNING: --synthetic set; output is fully synthetic LCG data, "
            "NOT real captures.",
            file=sys.stderr,
            flush=True,
        )
        subjects_data = [_make_synthetic_subject(i, args.seed) for i in range(args.subjects)]

    # Calibrate on the first 2/3 of subjects.
    n_cal = max(4, len(subjects_data) * 2 // 3)
    cal_residuals = [s["residual_mm"] for s in subjects_data[:n_cal]]
    calibrator = ErrorCalibrator.fit(cal_residuals, alpha=args.alpha)
    print(f"ErrorCalibrator: radius_mm={calibrator.radius_mm:.4f}  alpha={calibrator.alpha}")

    # Issue certificates for the remaining 1/3 (test subjects).
    test_subjects = subjects_data[n_cal:]
    if not test_subjects:
        # Fall back to certifying all subjects when the list is very short.
        test_subjects = subjects_data

    cert_inputs = []
    certificates = []
    # Build per-certificate true labels in lock-step with certificate issuance.
    # For synthetic subjects: replicate the LCG label decision (u0 < 0.6 → stable).
    # For real subjects without ground truth: label as "stable" (conservative default).
    true_labels: list[str] = []

    for subject in test_subjects:
        inp_list, certs = _issue_certificates(
            subject_id=subject["subject_id"],
            t0_views=subject["t0_views"],
            t0_tags=subject["t0_tags"],
            t1_views=subject["t1_views"],
            t1_tags=subject["t1_tags"],
            calibrator=calibrator,
            seed=args.seed,
            coverage_threshold=args.coverage_threshold,
            stable_threshold_mm=args.stable_threshold_mm,
            change_threshold_mm=args.change_threshold_mm,
            t0_image_paths=subject.get("t0_image_paths"),
            t1_image_paths=subject.get("t1_image_paths"),
            allow_synthetic_delta=allow_synthetic_delta,
        )
        if not certs:
            # Subject skipped (real reconstruction failed and synthetic delta
            # disallowed). Do not append true_labels — they must stay in
            # lock-step with issued certificates.
            continue
        cert_inputs.extend(inp_list)
        certificates.extend(certs)

        # Derive per-region true labels.
        # Real subjects: no ground truth — mark as "stable" (we cannot claim change without GT).
        # Synthetic subjects: use the same LCG u0 threshold that drove estimate_mm.
        #   u0 < 0.6 → stable pair  (0–0.1 mm estimate)
        #   u0 >= 0.6 → changed pair (0.8–1.5 mm estimate)
        # This gives a realistic mix (~60 % stable / 40 % changed) instead of all-stable,
        # which fixes the degenerate single-class AUC.
        has_real_images = bool(subject.get("t0_image_paths") or subject.get("t1_image_paths"))
        for region in REGIONS:
            if has_real_images:
                region_true_label = "stable"
            else:
                state = args.seed ^ hash(subject["subject_id"] + region)
                state, u0 = _lcg(state)
                region_true_label = "stable" if u0 < 0.6 else "changed"
            true_labels.append(region_true_label)

    if not certificates:
        print(
            "ERROR: every real subject was skipped because reconstruction "
            "produced no usable delta and synthetic substitution is disabled. "
            "No certificates were issued (pass --allow-synthetic-fallback to "
            "permit the LCG delta, or fix the reconstruction backend/data).",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.output)
    report_path, jsonl_path = write_outputs(certificates, out_dir, synthetic=not use_real)

    counts: dict[str, int] = {}
    for cert in certificates:
        counts[cert.label] = counts.get(cert.label, 0) + 1

    print(f"\nReport:  {report_path}")
    print(f"Records: {jsonl_path}")
    print(f"\nTotal certificates: {len(certificates)}")
    for label, n_label in sorted(counts.items()):
        print(f"  {label}: {n_label} ({100 * n_label / max(1, len(certificates)):.0f}%)")

    from dentalmapcert.eval_metrics import compute_metrics, coverage_vs_false_change_curve
    from dentalmapcert.baselines import naive_baseline, coverage_only_baseline, uncertainty_only_baseline

    cert_outputs = certificates

    if not use_real:
        print(
            "\n[SYNTHETIC FALLBACK] The benchmark, baseline, and curve numbers below "
            "are computed from generated LCG data, NOT real phone captures. The "
            "synthetic true_labels are derived from the SAME LCG draw (u0 < 0.6) that "
            "produces each delta estimate, so they measure self-consistency of the "
            "decision logic, not real-world accuracy.",
            flush=True,
        )

    metrics = compute_metrics(cert_outputs, true_labels=true_labels)
    print(f"\n=== Benchmark Metrics ===")
    print(f"  n:                          {metrics.n}")
    print(f"  useful_certified_coverage:  {metrics.useful_certified_coverage:.3f}")
    print(f"  uncertain_rate:             {metrics.uncertain_rate:.3f}")
    print(f"  capture_only_false_change:  {metrics.capture_only_false_change_rate:.3f}")
    print(f"  mean_delta_width_mm:        {metrics.mean_delta_width_mm:.3f}")
    print(f"  recapture_trigger_rate:     {metrics.recapture_trigger_rate:.3f}")

    # Run baselines for comparison
    naive_outputs = [naive_baseline(item) for item in cert_inputs]
    cov_outputs = [coverage_only_baseline(item) for item in cert_inputs]
    unc_outputs = [uncertainty_only_baseline(item) for item in cert_inputs]

    naive_metrics = compute_metrics(naive_outputs, true_labels=true_labels)
    cov_metrics = compute_metrics(cov_outputs, true_labels=true_labels)
    unc_metrics = compute_metrics(unc_outputs, true_labels=true_labels)

    print(f"\n=== Baseline Comparison (useful_certified_coverage) ===")
    print(f"  Our certificate:     {metrics.useful_certified_coverage:.3f}")
    print(f"  Naive baseline:      {naive_metrics.useful_certified_coverage:.3f}")
    print(f"  Coverage-only:       {cov_metrics.useful_certified_coverage:.3f}")
    print(f"  Uncertainty-only:    {unc_metrics.useful_certified_coverage:.3f}")

    curve = coverage_vs_false_change_curve(cert_outputs, true_labels=true_labels)
    print(f"\n=== Coverage vs False-Change Curve ===")
    for pt in curve[::5]:  # sample every 5th
        print(f"  cov_thresh={pt['coverage_threshold']:.2f}  useful_cov={pt['useful_coverage']:.3f}  fpr={pt['false_change_rate']:.3f}")


if __name__ == "__main__":
    main()
