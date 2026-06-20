#!/usr/bin/env python3
"""Thorough surface-certificate evaluation with ablations (Poseidon3D, oracle).

Validates the conformal surface-change certificate on real arches and quantifies
the honest dependence on reconstruction noise — the lever that decides usability.

Two estimators are compared:
  * raw       — mean per-point displacement norm (the original measurement). This
                *rectifies* zero-mean reconstruction noise into a positive bias
                ~2.3*sigma, so the conformal radius grows ~linearly with noise and
                sensitivity collapses.
  * debiased  — toothprint.surface.surface_displacement: noise-power subtraction
                sqrt(max(0, mean||v||^2 - floor)), floor estimated from stable
                pairs. Removes the bias, so the certificate keeps sensitivity as
                noise grows.

The de-biased gain depends on the noise being spatially *incoherent*; a
correlated-noise ablation (corr in [0,1]) shows the gain shrinks for realistic,
spatially smooth reconstruction error. Change is SYNTHETIC (uniform outward
displacement); reconstruction error is modelled noise.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dentalmapcert.calibration import ErrorCalibrator
from dentalmapcert.dataset_loaders import Poseidon3DLoader, load_poseidon3d_points
from toothprint.surface.error import noise_floor_sq, surface_displacement

OUT = Path("/home/krishi/personal-projects/toothprint/evaluation/results/surface.json")
DATA = "data/poseidon3d/extracted/data"
ST, CT = 0.35, 0.75  # stable / change decision thresholds (mm)


def displace(pts, mm):
    if mm == 0:
        return pts.copy()
    c = pts.mean(0)
    u = (pts - c) / np.clip(np.linalg.norm(pts - c, axis=1, keepdims=True), 1e-9, None)
    return pts + u * mm


def correlated_noise(points, sigma, corr, rng, n_modes=6):
    """Reconstruction noise of per-axis std ~sigma; ``corr`` in [0,1] is the
    fraction of variance that is spatially smooth (low-frequency, does not average
    out across points). corr=0 -> independent; corr=1 -> fully coherent."""
    indep = rng.normal(0, 1, points.shape)
    if corr <= 0:
        return sigma * indep
    P = points - points.mean(0)
    scale = float(np.linalg.norm(P, axis=1).mean()) + 1e-9
    smooth = np.zeros_like(points)
    for ax in range(3):
        f = np.zeros(points.shape[0])
        for _ in range(n_modes):
            k = rng.normal(0, 1.0 / scale, 3)
            f += np.sin(P @ k + rng.uniform(0, 2 * np.pi))
        smooth[:, ax] = f
    smooth /= (smooth.std() + 1e-9)
    return sigma * (np.sqrt(1 - corr) * indep + np.sqrt(corr) * smooth)


def noisy(pts, sigma, corr, rng):
    return pts + correlated_noise(pts, sigma, corr, rng)


def raw_measure(t0n, t1n):
    return float(np.linalg.norm(t1n - t0n, axis=1).mean())


def certify(measured, cal):
    lo, hi = cal.interval(measured)
    lo = max(0.0, lo)
    if lo >= CT:
        return "changed"
    if hi <= ST:
        return "stable"
    return "uncertain"


def run(surfaces, *, noise, alpha, reps, changes, corr=0.0, estimator="debiased"):
    rng = np.random.default_rng(0)
    stable = [(noisy(s, noise, corr, rng), noisy(s, noise, corr, rng))
              for s in surfaces for _ in range(reps)]
    floor = noise_floor_sq(stable) if estimator == "debiased" else 0.0
    meas = (lambda a, b: surface_displacement(a, b, noise_floor_sq=floor)) \
        if estimator == "debiased" else raw_measure
    cal = ErrorCalibrator.fit([meas(a, b) for a, b in stable], alpha=alpha)
    curve = []
    for ch in changes:
        n = cc = sc = 0
        for s in surfaces:
            t1 = displace(s, ch)
            for _ in range(reps):
                lab = certify(meas(noisy(s, noise, corr, rng), noisy(t1, noise, corr, rng)), cal)
                n += 1; cc += lab == "changed"; sc += lab == "stable"
        rate = cc / n
        curve.append({"change_mm": ch, "changed_rate": rate, "stable_rate": sc / n,
                      "recall": rate if ch >= CT else None, "fpr": rate if ch == 0 else None})
    return {"radius_mm": cal.radius_mm, "estimator": estimator, "corr": corr, "curve": curve}


def _rec1(block):
    return next(x["changed_rate"] for x in block["curve"] if x["change_mm"] == 1.0)


def _fpr(block):
    return next(x["changed_rate"] for x in block["curve"] if x["change_mm"] == 0.0)


def main():
    loader = Poseidon3DLoader(DATA)
    recs = [r for r in loader.records() if r.mesh_path and Path(r.mesh_path).exists()][:8]
    surfaces = [load_poseidon3d_points(r, n_points=2000, seed=0) for r in recs]
    res = {"dataset": "poseidon3d", "n_meshes": len(surfaces), "estimator": "debiased",
           "ablations": {}, "baseline_raw": {}, "correlated": {}}
    changes = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    noises = [0.03, 0.05, 0.10, 0.20, 0.40, 0.84]  # 0.84 = our Gaussian-Splatting photo-recon

    print("[main] de-biased, recon noise 0.05mm (IOS), alpha 0.1", flush=True)
    res["main"] = run(surfaces, noise=0.05, alpha=0.1, reps=12, changes=changes)
    for r in res["main"]["curve"]:
        print(f"  change {r['change_mm']}mm: changed={r['changed_rate']:.2f} stable={r['stable_rate']:.2f}", flush=True)

    print("[noise] de-biased vs raw, independent noise", flush=True)
    print(f"  {'noise':>5} {'raw_radius':>10} {'raw_rec@1':>9} {'db_radius':>9} {'db_rec@1':>8} {'db_fpr':>6}", flush=True)
    for nz in noises:
        db = run(surfaces, noise=nz, alpha=0.1, reps=12, changes=changes, estimator="debiased")
        raw = run(surfaces, noise=nz, alpha=0.1, reps=12, changes=changes, estimator="raw")
        res["ablations"][f"noise_{nz}"] = db
        res["baseline_raw"][f"noise_{nz}"] = raw
        print(f"  {nz:>5} {raw['radius_mm']:>10.3f} {_rec1(raw):>9.2f} "
              f"{db['radius_mm']:>9.3f} {_rec1(db):>8.2f} {_fpr(db):>6.3f}", flush=True)

    print("[correlated] de-biased recall@1mm vs noise correlation (noise 0.20mm)", flush=True)
    for corr in [0.0, 0.5, 0.9, 1.0]:
        block = run(surfaces, noise=0.20, alpha=0.1, reps=12, changes=changes, corr=corr)
        res["correlated"][f"corr_{corr}"] = block
        print(f"  corr {corr}: radius={block['radius_mm']:.3f} recall@1mm={_rec1(block):.2f} "
              f"fpr={_fpr(block):.3f}", flush=True)

    print("[alpha] FPR <= alpha (de-biased, noise 0.05)", flush=True)
    for a in [0.05, 0.1, 0.2]:
        block = run(surfaces, noise=0.05, alpha=a, reps=12, changes=changes)
        res["ablations"][f"alpha_{a}"] = block
        print(f"  alpha {a}: fpr={_fpr(block):.3f} (<= {a}? {_fpr(block) <= a})", flush=True)

    OUT.write_text(json.dumps(res, indent=1))
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
