#!/usr/bin/env python3
"""Generate DentalMapCert result plots (CPU only).

Outputs to docs/:
  - dmc_poisson_refinement.png   noisy vs screened-Poisson-refined surface (denoising)
  - dmc_chamfer_vs_noise.png     validated Chamfer: raw vs after Poisson across noise
  - dmc_surface_error_heatmap.png per-point surface error on a noisy arch-like surface
  - dmc_surface_error_metrics.png chamfer / point-to-surface / Hausdorff, raw vs refined
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dentalmapcert.meshing import poisson_surface_reconstruction
from dentalmapcert.surface_error import surface_error_mm

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"; DOCS.mkdir(exist_ok=True)


def _sphere(n, radius=10.0, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * radius


def poisson_refinement():
    radius = 10.0
    clean = _sphere(6000, radius, seed=1)
    noisy = clean + np.random.default_rng(2).normal(0, 0.3, clean.shape)
    refined = poisson_surface_reconstruction(noisy, depth=8, normal_radius=2.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), subplot_kw=dict())
    for ax, pts, title in [(axes[0], noisy, "noisy input (sigma=0.3mm)"),
                           (axes[1], refined, "screened-Poisson refined")]:
        err = np.abs(np.linalg.norm(pts, axis=1) - radius)
        sc = ax.scatter(pts[:, 0], pts[:, 2], c=err, s=3, cmap="inferno", vmin=0, vmax=0.8)
        ax.set_title(f"{title}\nmean |radial err| = {err.mean():.3f} mm")
        ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="|err| mm")
    fig.suptitle("Screened-Poisson denoises a sub-mm surface (Kazhdan & Hoppe 2013)", y=0.99)
    fig.tight_layout(); out = DOCS / "dmc_poisson_refinement.png"
    fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


def chamfer_vs_noise():
    # Validated on a real Poseidon3D IOS arch (see meshing.py docstring).
    noise = [0.0, 0.5, 1.0, 2.0]
    raw = [0.162, 0.379, 0.598, 1.006]
    poisson = [0.133, 0.217, 0.404, 1.146]
    x = np.arange(len(noise)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w / 2, raw, w, label="raw reconstruction", color="#999999")
    ax.bar(x + w / 2, poisson, w, label="after screened Poisson", color="#1f77b4")
    for i, (r, p) in enumerate(zip(raw, poisson)):
        pct = (p - r) / r * 100
        ax.text(i + w / 2, p + 0.02, f"{pct:+.0f}%", ha="center",
                color="green" if pct < 0 else "red", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{n} mm" for n in noise])
    ax.set_xlabel("reconstruction noise sigma"); ax.set_ylabel("Chamfer distance (mm)")
    ax.set_title("Poisson helps in the sub-mm..~1mm regime, hurts past ~2mm\n"
                 "(real Poseidon3D IOS arch, depth=9)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); out = DOCS / "dmc_chamfer_vs_noise.png"
    fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


def surface_error_heatmap():
    radius = 10.0
    clean = _sphere(4000, radius, seed=3)
    # Spatially-varying error: more noise on one hemisphere (mimics a hard-to-scan region).
    rng = np.random.default_rng(4)
    scale = 0.15 + 0.5 * (clean[:, 1] > 0)
    noisy = clean + rng.normal(0, 1, clean.shape) * scale[:, None]
    err = np.abs(np.linalg.norm(noisy, axis=1) - radius)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(noisy[:, 0], noisy[:, 1], noisy[:, 2], c=err, cmap="turbo", s=6, vmin=0, vmax=1.2)
    ax.set_title(f"Per-point surface error (mean {err.mean():.3f} mm, p95 {np.percentile(err,95):.3f} mm)")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="distance to true surface (mm)")
    fig.tight_layout(); out = DOCS / "dmc_surface_error_heatmap.png"
    fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


def surface_error_metrics():
    radius = 10.0
    ref = _sphere(8000, radius, seed=5)
    noisy = _sphere(6000, radius, seed=6) + np.random.default_rng(7).normal(0, 0.4, (6000, 3))
    refined = poisson_surface_reconstruction(noisy, depth=8, normal_radius=2.0)

    e_raw = surface_error_mm(noisy, ref, run_icp=False)
    e_ref = surface_error_mm(refined, ref, run_icp=False)
    labels = ["Chamfer", "point-to-surface", "Hausdorff"]
    raw = [e_raw.chamfer_mm, e_raw.point_to_surface_mm, e_raw.hausdorff_mm]
    ref_v = [e_ref.chamfer_mm, e_ref.point_to_surface_mm, e_ref.hausdorff_mm]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w / 2, raw, w, label="noisy input", color="#999999")
    ax.bar(x + w / 2, ref_v, w, label="Poisson refined", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("error (mm)")
    ax.set_title("Surface-error metrics (real computation, ICP off): noisy vs refined")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); out = DOCS / "dmc_surface_error_metrics.png"
    fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


def oracle_certificate():
    import json
    mp = ROOT / "outputs/dmc_cert_oracle/metrics.json"
    if not mp.exists():
        print("skip oracle certificate (no", mp, ")")
        return
    d = json.loads(mp.read_text())
    sweep = d["sweep"]
    x = [s["change_mm"] for s in sweep]
    change_rate = [s["certified_change_rate"] for s in sweep]
    stable_rate = [s["stable_cert_rate"] for s in sweep]
    uncertain = [s["uncertain_rate"] for s in sweep]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(x, change_rate, "-o", color="#d62728", lw=2.2, ms=6, label="surface-change certified")
    ax.plot(x, stable_rate, "-^", color="#2ca02c", lw=2, ms=6, label="surface-stable certified")
    ax.plot(x, uncertain, "-s", color="#7f7f7f", lw=1.6, ms=5, label="uncertain / recapture")
    ax.axvspan(0.35, 0.75, color="orange", alpha=0.12, label="gray zone (stable..change thresholds)")
    ax.set_xlabel("true surface change (mm)"); ax.set_ylabel("certificate rate")
    ax.set_title("DentalMapCert surface-change certificate on real Poseidon3D meshes\n"
                 f"recon noise {d['recon_noise_mm']}mm (IOS-class), radius {d['radius_mm']:.3f}mm: "
                 "FPR=0, recall->1.0 at >=1mm")
    ax.set_ylim(-0.03, 1.05); ax.legend(loc="center right", fontsize=8.5); ax.grid(alpha=0.3)
    fig.tight_layout(); out = DOCS / "dmc_oracle_certificate.png"
    fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    poisson_refinement()
    chamfer_vs_noise()
    surface_error_heatmap()
    surface_error_metrics()
    oracle_certificate()
