#!/usr/bin/env python3
"""DentalMapCert surface-change certificate validated on real meshes (oracle).

Mirrors the DCC certificate validation: it isolates the *certificate* (the
decision rule) from the reconstruction front-end. Real Poseidon3D dental
surfaces are the geometry; a controlled metric surface change (uniform outward
displacement of the surface by `change_mm`, modelling swelling / recession) is
the true-positive signal; reconstruction error is modelled as calibrated
Gaussian surface noise and absorbed by the conformal error radius.

The question: given a calibrated reconstruction-error radius, does the certificate
flag a real sub-mm surface change while never flagging reconstruction noise?

    python scripts/run_dmc_certificate_oracle.py --data data/poseidon3d/extracted/data \
        --output outputs/dmc_cert_oracle --recon-noise-mm 0.1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dentalmapcert.calibration import ErrorCalibrator
from dentalmapcert.certificate import CertificateInput, decide_surface_change
from dentalmapcert.dataset_loaders import Poseidon3DLoader, load_poseidon3d_points


def _displace_outward(points: np.ndarray, change_mm: float) -> np.ndarray:
    """Move every point radially outward from the centroid by ``change_mm``.

    A uniform outward normal displacement models surface swelling/recession; it
    is not a rigid motion, so it survives registration and is a real change.
    """
    if change_mm == 0.0:
        return points.copy()
    c = points.mean(axis=0)
    rel = points - c
    norm = np.linalg.norm(rel, axis=1, keepdims=True)
    unit = rel / np.clip(norm, 1e-9, None)
    return points + unit * change_mm


def _measure_change(t0: np.ndarray, t1: np.ndarray, rng, noise_mm: float) -> float:
    """Reconstruct both timepoints (GT + calibrated noise) and measure the change.

    Oracle measurement: with perfect reconstruction the two timepoints are in
    correspondence, so the surface change is the mean per-point displacement.
    This is the 3-D analogue of DCC's accurate-landmark measurement and avoids
    the sampling-density floor of a nearest-neighbour point-to-surface distance.
    """
    recon_t0 = t0 + rng.normal(0, noise_mm, t0.shape)
    recon_t1 = t1 + rng.normal(0, noise_mm, t1.shape)
    return float(np.linalg.norm(recon_t1 - recon_t0, axis=1).mean())


def main() -> int:
    p = argparse.ArgumentParser(description="DMC oracle surface-change certificate")
    p.add_argument("--data", default="data/poseidon3d/extracted/data")
    p.add_argument("--output", default="outputs/dmc_cert_oracle")
    p.add_argument("--n-meshes", type=int, default=6)
    p.add_argument("--n-points", type=int, default=2000)
    p.add_argument("--recon-noise-mm", type=float, default=0.1,
                   help="Modelled reconstruction surface-noise std (mm)")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--reps", type=int, default=12, help="Noise reps per mesh per condition")
    p.add_argument("--sweep", default="0.0,0.2,0.4,0.6,0.8,1.0,1.5",
                   help="True surface-change magnitudes (mm)")
    args = p.parse_args()

    root = Path(args.data)
    if not root.exists():
        print(f"ERROR: Poseidon3D data not found at {root}", file=sys.stderr)
        return 1

    loader = Poseidon3DLoader(str(root))
    records = [r for r in loader.records() if r.mesh_path and Path(r.mesh_path).exists()]
    records = records[: args.n_meshes]
    if not records:
        print(f"ERROR: no meshes under {root}", file=sys.stderr)
        return 1
    print(f"Meshes: {len(records)}   recon-noise: {args.recon_noise_mm} mm")

    rng = np.random.default_rng(0)
    surfaces = [load_poseidon3d_points(r, n_points=args.n_points, seed=0) for r in records]

    # --- Calibrate the error radius on stable (no-change) reconstructions. ---
    residuals = []
    for s in surfaces:
        for _ in range(args.reps):
            residuals.append(_measure_change(s, s.copy(), rng, args.recon_noise_mm))
    calibrator = ErrorCalibrator.fit(residuals, alpha=args.alpha)
    print(f"Calibrated error radius: {calibrator.radius_mm:.4f} mm "
          f"(from {len(residuals)} stable reps)")

    # --- Sweep the true change magnitude; certify each. ---
    sweep = [float(x) for x in args.sweep.split(",")]
    curve = []
    print(f"\n  {'change_mm':>9} {'recall':>8} {'FPR':>8} {'stable_cert':>12} {'uncertain':>10}")
    for change in sweep:
        n = certified_change = certified_stable = uncertain = 0
        for s in surfaces:
            t1_template = _displace_outward(s, change)
            for _ in range(args.reps):
                measured = _measure_change(s, t1_template, rng, args.recon_noise_mm)
                lo, hi = calibrator.interval(measured)
                inp = CertificateInput(
                    surface_region_id="r", capture_id_t0="t0", capture_id_t1="t1",
                    coverage_score_t0=1.0, coverage_score_t1=1.0,
                    error_interval_mm_t0=calibrator.interval(0.0),
                    error_interval_mm_t1=calibrator.interval(0.0),
                    delta_interval_mm=(round(max(0.0, lo), 6), round(hi, 6)),
                )
                label = decide_surface_change(inp).label
                n += 1
                if label == "surface change certified":
                    certified_change += 1
                elif label == "surface stable certified":
                    certified_stable += 1
                else:
                    uncertain += 1
        is_change = change >= 0.75  # change_threshold default
        rate = certified_change / n
        row = {"change_mm": change,
               "recall": rate if is_change else None,
               "fpr": rate if not is_change else None,
               "certified_change_rate": rate,
               "stable_cert_rate": certified_stable / n,
               "uncertain_rate": uncertain / n}
        curve.append(row)
        recall = rate if is_change else float("nan")
        fpr = rate if not is_change else float("nan")
        print(f"  {change:>9.2f} {recall:>8.3f} {fpr:>8.3f} "
              f"{certified_stable / n:>12.3f} {uncertain / n:>10.3f}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": "poseidon3d", "n_meshes": len(records),
               "recon_noise_mm": args.recon_noise_mm, "alpha": args.alpha,
               "radius_mm": calibrator.radius_mm, "sweep": curve}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Metrics: {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
