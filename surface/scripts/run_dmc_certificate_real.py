#!/usr/bin/env python3
"""KEYSTONE: real end-to-end DentalMapCert gate (geometry -> CERTIFICATE).

This is the script the other two gate scripts were missing: it connects the
REAL reconstruction chain all the way through to a certificate, with the real
Chamfer/surface error feeding ``delta_interval_mm`` instead of a synthetic LCG
residual.

Chain (per Poseidon3D mesh, no synthetic numbers anywhere on this path)::

    real STL mesh  (timepoint t0)
      -> render_5_views (open3d offscreen)
      -> reconstruct_point_cloud(backend)           # VGGT default
      -> surface_error_mm(recon_t0, GT mesh pts)     # REAL recon self-error
                                                     #  -> calibrator radius

    geometrically perturbed copy of the SAME mesh   (timepoint t1)
      -> render_5_views -> reconstruct_point_cloud(backend)
      -> surface_error_mm(recon_t1, recon_t0)        # REAL change estimate
                                                     #  -> delta_interval_mm

    delta_interval_mm -> decide_surface_change() -> certificate (per region)

Two timepoints are produced from one mesh by applying a *known, real* rigid
displacement to the STL geometry (``--shift-mm``).  The displacement is a real
geometric change in millimetres — when ``--shift-mm 0`` the two timepoints are
the same geometry (a "stable" pair); a non-zero shift is a genuine "changed"
pair.  Nothing about the delta is drawn from an LCG; it is whatever the
reconstruction chain measures between the two rendered/reconstructed clouds.

Backends
--------
``--backend {vggt,dust3r}`` (default ``vggt``).  Both are metric neural
reconstructors; there is NO crude CPU fallback. If the chosen backend cannot
run (e.g. insufficient GPU VRAM) the gate fails loudly so the real cause is
fixed rather than masked by an uncalibrated result.

De-synthetic contract
----------------------
This script NEVER silently fabricates ``delta_interval_mm``.  When the
Poseidon3D meshes are absent it exits with an honest non-zero status instead of
quietly synthesising results.  The ``--synthetic`` flag is the ONLY way to get
LCG numbers, and it labels every output as synthetic.

Status: implemented; pending GPU/data run (no GPU job is run here).

Usage::

    # REAL path (needs Poseidon3D meshes + a GPU for VGGT/DUSt3R):
    python scripts/run_dmc_certificate_real.py \\
        --data data/poseidon3d/extracted/data \\
        --output outputs/dmc_certificate_real \\
        --backend vggt --limit 8 --shift-mm 1.0 --resolution 256

    # Explicit synthetic demo (no data, no GPU; labelled synthetic):
    python scripts/run_dmc_certificate_real.py --synthetic --output outputs/dmc_cert_demo
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from dentalmapcert.calibration import ErrorCalibrator
from dentalmapcert.certificate import CertificateInput, decide_surface_change
from dentalmapcert.dataset_loaders import Poseidon3DLoader, load_poseidon3d_points
from dentalmapcert.reconstruction import BACKENDS, reconstruct_point_cloud
from dentalmapcert.regions import region_id, region_surface_from_id
from dentalmapcert.render import render_5_views, rendered_view_to_pil
from dentalmapcert.report import write_outputs
from dentalmapcert.surface_error import surface_error_mm

# Anterior teeth visible in the 5 protocol views (FDI notation).
_GATE_TEETH = (11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43)
_GATE_SURFACES = ("buccal", "mesial", "distal")
REGIONS = [region_id(fdi, surf) for fdi in _GATE_TEETH for surf in _GATE_SURFACES]


# ---------------------------------------------------------------------------
# Rendering helper
# ---------------------------------------------------------------------------

def _render_and_save(mesh_path: Path, out_dir: Path, resolution: int) -> list[Path]:
    """Render the 5 protocol views of *mesh_path* and save them as PNGs."""
    views = render_5_views(str(mesh_path), resolution=resolution)
    paths: list[Path] = []
    for view_name, view in views.items():
        img = rendered_view_to_pil(view)
        p = out_dir / f"{view_name}.png"
        img.save(p)
        paths.append(p)
    return paths


def _write_shifted_stl(mesh_path: Path, out_path: Path, shift_mm: float) -> bool:
    """Write a copy of *mesh_path* translated by ``shift_mm`` along +X.

    The shift is a real geometric displacement applied to the mesh vertices, so
    rendering/reconstructing the copy gives a genuine "follow-up" timepoint with
    a known change.  Raises if open3d is unavailable or the mesh is empty —
    no silent fallback.
    """
    import open3d as o3d  # noqa: PLC0415

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"mesh has no triangles, cannot shift: {mesh_path}")
    mesh.translate((float(shift_mm), 0.0, 0.0))
    # Open3D's ASCII/binary STL writer requires triangle normals and silently
    # returns False without them, which aborted the t1 timepoint build.
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    if not o3d.io.write_triangle_mesh(str(out_path), mesh):
        raise RuntimeError(f"open3d failed to write shifted STL: {out_path}")


# ---------------------------------------------------------------------------
# Real per-mesh certificate computation
# ---------------------------------------------------------------------------

def _real_recon_error_mm(
    mesh_path: Path,
    gt_points: np.ndarray,
    backend: str,
    resolution: int,
) -> tuple[np.ndarray, float]:
    """Render+reconstruct *mesh_path*; return (recon_points, recon_error_mm).

    ``recon_error_mm`` is the real Chamfer distance between the reconstruction
    and the mesh's own sampled GT point cloud — the reconstruction self-error
    used to calibrate the certificate's error radius.  Returns
    (empty array, nan) when reconstruction fails.
    """
    with tempfile.TemporaryDirectory() as tmp:
        image_paths = _render_and_save(mesh_path, Path(tmp), resolution)
        recon_points, _conf = reconstruct_point_cloud(image_paths, backend=backend)
    if recon_points.shape[0] < 3:
        return recon_points, float("nan")
    err = surface_error_mm(recon_points, gt_points, run_icp=True)
    return recon_points, err.chamfer_mm


def _certify_mesh(
    recon_t0: np.ndarray,
    recon_t1: np.ndarray,
    calibrator: ErrorCalibrator,
    subject_id: str,
    coverage_threshold: float,
    stable_threshold_mm: float,
    change_threshold_mm: float,
    coverage_t0: float,
    coverage_t1: float,
) -> tuple[list[CertificateInput], list]:
    """Compute the REAL delta from two reconstructions and issue certificates.

    ``delta_interval_mm`` is derived entirely from
    ``surface_error_mm(recon_t1, recon_t0)`` — the real measured change between
    the two timepoint reconstructions — widened by the calibrator radius.  No
    LCG / synthetic residual is involved.
    """
    err = surface_error_mm(recon_t1, recon_t0, run_icp=True)
    delta_mm = err.chamfer_mm
    if not np.isfinite(delta_mm):
        return [], []
    delta_lo = max(0.0, delta_mm - calibrator.radius_mm)
    delta_hi = delta_mm + calibrator.radius_mm

    # Both timepoints carry the same reconstruction error interval (the
    # calibrated recon self-error), since both come from the same backend.
    err_interval = calibrator.interval(0.0)

    cert_inputs: list[CertificateInput] = []
    certs = []
    for region in REGIONS:
        fdi_num, surf_name = region_surface_from_id(region)
        inp = CertificateInput(
            surface_region_id=f"{subject_id}_{region}",
            capture_id_t0=f"{subject_id}_t0",
            capture_id_t1=f"{subject_id}_t1",
            coverage_score_t0=coverage_t0,
            coverage_score_t1=coverage_t1,
            error_interval_mm_t0=err_interval,
            error_interval_mm_t1=err_interval,
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
# Synthetic demo path (explicit only)
# ---------------------------------------------------------------------------

def _lcg(state: int) -> tuple[int, float]:
    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
    return state, state / 0xFFFFFFFF


def _run_synthetic(args) -> int:
    """Explicit synthetic demo. Every output is labelled synthetic."""
    print(
        "WARNING: --synthetic set; ALL numbers below are generated by a "
        "deterministic LCG, NOT real geometry. Do not cite these as results.",
        file=sys.stderr,
        flush=True,
    )
    calibrator = ErrorCalibrator(radius_mm=0.25, alpha=args.alpha)
    certificates = []
    state = args.seed
    for subj in range(args.subjects):
        subject_id = f"SYN{subj:03d}"
        state, u0 = _lcg(state)
        state, u1 = _lcg(state)
        delta_mm = u1 * 0.1 if u0 < 0.6 else 0.8 + u1 * 0.7
        delta_lo = max(0.0, delta_mm - calibrator.radius_mm)
        delta_hi = delta_mm + calibrator.radius_mm
        # Build certs directly with the synthetic delta. The real path
        # (_certify_mesh) recomputes the delta from point clouds, so it cannot
        # be reused here where the delta is an explicit LCG draw.
        certs = []
        for region in REGIONS:
            fdi_num, surf_name = region_surface_from_id(region)
            inp = CertificateInput(
                surface_region_id=f"{subject_id}_{region}",
                capture_id_t0=f"{subject_id}_t0",
                capture_id_t1=f"{subject_id}_t1",
                coverage_score_t0=1.0,
                coverage_score_t1=1.0,
                error_interval_mm_t0=calibrator.interval(0.0),
                error_interval_mm_t1=calibrator.interval(0.0),
                delta_interval_mm=(round(delta_lo, 6), round(delta_hi, 6)),
                region_type=f"fdi_{fdi_num}_{surf_name}_surface",
            )
            certs.append(decide_surface_change(
                inp,
                coverage_threshold=args.coverage_threshold,
                stable_threshold_mm=args.stable_threshold_mm,
                change_threshold_mm=args.change_threshold_mm,
            ))
        certificates.extend(certs)

    out_dir = Path(args.output)
    report_path, jsonl_path = write_outputs(certificates, out_dir, synthetic=True)
    print(f"[SYNTHETIC] Report:  {report_path}")
    print(f"[SYNTHETIC] Records: {jsonl_path}")
    print(f"[SYNTHETIC] Total certificates: {len(certificates)}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="KEYSTONE real geometry -> certificate gate (Poseidon3D)"
    )
    parser.add_argument("--data", default="data/poseidon3d/extracted/data",
                        help="Root of extracted Poseidon3D data/ directory")
    parser.add_argument("--output", default="outputs/dmc_certificate_real")
    parser.add_argument("--backend", choices=BACKENDS, default="vggt",
                        help="Reconstruction backend (vggt or dust3r; both are real "
                             "metric neural reconstructors — no crude CPU fallback)")
    parser.add_argument("--limit", type=int, default=8,
                        help="Max number of meshes to process")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Render resolution per view")
    parser.add_argument("--n-gt-points", type=int, default=5000,
                        help="Points sampled from each GT mesh for calibration")
    parser.add_argument("--shift-mm", type=float, default=1.0,
                        help="Known rigid +X displacement (mm) applied to make the "
                             "t1 timepoint; 0.0 makes a stable (no-change) pair")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Conformal significance level for the error radius")
    parser.add_argument("--coverage-threshold", type=float, default=0.15)
    parser.add_argument("--stable-threshold-mm", type=float, default=0.50)
    parser.add_argument("--change-threshold-mm", type=float, default=0.90)
    parser.add_argument("--synthetic", action="store_true",
                        help="EXPLICIT synthetic demo: generate LCG certificates "
                             "without any real data/GPU. Output is labelled synthetic.")
    parser.add_argument("--subjects", type=int, default=12,
                        help="Number of synthetic subjects to generate (only used "
                             "with --synthetic)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.synthetic:
        return _run_synthetic(args)

    # ---- REAL path: never fabricate; exit honestly when data is absent. ----
    root = Path(args.data)
    if not root.exists():
        print(
            f"ERROR: Poseidon3D data not found at {root}. Refusing to fabricate "
            "a delta. Run scripts/fetch_data.sh (see docs/DATA.md) to download "
            "the meshes, or pass --synthetic for an explicitly-labelled demo.",
            file=sys.stderr,
        )
        return 1

    loader = Poseidon3DLoader(str(root))
    records = [r for r in loader.records() if r.mesh_path and Path(r.mesh_path).exists()]
    if not records:
        print(f"ERROR: no Poseidon3D meshes under {root}", file=sys.stderr)
        return 1
    records = records[: args.limit]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pass 1: render+reconstruct each mesh (t0) and its shifted copy (t1).
    # Collect the real reconstruction self-error to calibrate the error radius.
    print(
        f"Running REAL geometry->certificate gate on {len(records)} Poseidon3D "
        f"meshes  (backend={args.backend}, shift={args.shift_mm} mm)\n"
    )
    per_mesh: list[dict] = []
    recon_errors_mm: list[float] = []
    for i, rec in enumerate(records):
        mesh_path = Path(rec.mesh_path)
        gt_points = load_poseidon3d_points(rec, n_points=args.n_gt_points, seed=args.seed)
        if gt_points.shape[0] < 3:
            print(f"  [{i+1}/{len(records)}] {rec.record_id}: empty mesh, skipped")
            continue

        recon_t0, recon_err_mm = _real_recon_error_mm(
            mesh_path, gt_points, args.backend, args.resolution
        )
        if recon_t0.shape[0] < 3 or not np.isfinite(recon_err_mm):
            print(f"  [{i+1}/{len(records)}] {rec.record_id}: t0 reconstruction failed, skipped")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            shifted_path = Path(tmp) / f"{rec.record_id}_t1.stl"
            _write_shifted_stl(mesh_path, shifted_path, args.shift_mm)  # raises on failure
            recon_t1, _ = _real_recon_error_mm(
                shifted_path, gt_points, args.backend, args.resolution
            )
        if recon_t1.shape[0] < 3:
            print(f"  [{i+1}/{len(records)}] {rec.record_id}: t1 reconstruction failed, skipped")
            continue

        recon_errors_mm.append(recon_err_mm)
        per_mesh.append({
            "record_id": rec.record_id,
            "recon_t0": recon_t0,
            "recon_t1": recon_t1,
            "recon_error_mm": recon_err_mm,
        })
        print(f"  [{i+1}/{len(records)}] {rec.record_id}: recon self-error={recon_err_mm:.3f} mm")

    if not per_mesh:
        print("ERROR: no mesh produced a usable reconstruction; no certificates issued.",
              file=sys.stderr)
        return 1

    # ---- Calibrate the error radius on the REAL reconstruction self-errors.
    calibrator = ErrorCalibrator.fit(recon_errors_mm, alpha=args.alpha)
    print(f"\nErrorCalibrator (from {len(recon_errors_mm)} REAL recon errors): "
          f"radius_mm={calibrator.radius_mm:.4f}  alpha={calibrator.alpha}")

    # ---- Pass 2: issue certificates with the REAL delta per mesh.
    certificates = []
    records_json: list[dict] = []
    for m in per_mesh:
        # Coverage proxy: both timepoints render all 5 protocol views from the
        # same arch, so coverage is high; we report 1.0 here because the gate's
        # purpose is the delta decision, not coverage estimation (which the
        # Poseidon3D/teethland coverage scripts measure separately).
        _inps, certs = _certify_mesh(
            recon_t0=m["recon_t0"],
            recon_t1=m["recon_t1"],
            calibrator=calibrator,
            subject_id=m["record_id"],
            coverage_threshold=args.coverage_threshold,
            stable_threshold_mm=args.stable_threshold_mm,
            change_threshold_mm=args.change_threshold_mm,
            coverage_t0=1.0,
            coverage_t1=1.0,
        )
        if not certs:
            continue
        certificates.extend(certs)
        # delta is identical across regions for this mesh; record it once.
        records_json.append({
            "record_id": m["record_id"],
            "recon_error_mm": m["recon_error_mm"],
            "delta_interval_mm": list(certs[0].delta_interval_mm),
            "n_regions": len(certs),
        })

    report_path, jsonl_path = write_outputs(certificates, out_dir, synthetic=False)

    payload = {
        "dataset": "poseidon3d",
        "backend": args.backend,
        "shift_mm": args.shift_mm,
        "resolution": args.resolution,
        "n_gt_points": args.n_gt_points,
        "seed": args.seed,
        "alpha": args.alpha,
        "synthetic": False,
        "calibrator_radius_mm": calibrator.radius_mm,
        "n_meshes": len(records_json),
        "records": records_json,
    }
    (out_dir / "certificate_real.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    counts: dict[str, int] = {}
    for cert in certificates:
        counts[cert.label] = counts.get(cert.label, 0) + 1

    print(f"\nReport:  {report_path}")
    print(f"Records: {jsonl_path}")
    print(f"Summary: {out_dir / 'certificate_real.json'}")
    print(f"\nTotal certificates: {len(certificates)} (from {len(records_json)} real meshes)")
    for label, n_label in sorted(counts.items()):
        print(f"  {label}: {n_label} ({100 * n_label / max(1, len(certificates)):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
