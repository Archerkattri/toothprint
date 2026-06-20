#!/usr/bin/env python3
"""Thorough surface-certificate evaluation with ablations (Poseidon3D, oracle).

Validates the conformal surface-change certificate on real arches: false-change
rate vs the change/stable decision, recall vs true displacement, and the
dependence on reconstruction noise (the lever that decides usability). Change is
SYNTHETIC (uniform outward displacement); reconstruction error is modelled noise.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dentalmapcert.calibration import ErrorCalibrator
from dentalmapcert.certificate import CertificateInput, decide_surface_change
from dentalmapcert.dataset_loaders import Poseidon3DLoader, load_poseidon3d_points

OUT = Path("/home/krishi/personal-projects/toothprint/evaluation/results/surface.json")
DATA = "data/poseidon3d/extracted/data"


def displace(pts, mm):
    if mm == 0:
        return pts.copy()
    c = pts.mean(0); rel = pts - c
    u = rel / np.clip(np.linalg.norm(rel, axis=1, keepdims=True), 1e-9, None)
    return pts + u * mm


def measure(t0, t1, rng, noise):
    return float(np.linalg.norm((t1 + rng.normal(0, noise, t1.shape)) - (t0 + rng.normal(0, noise, t0.shape)), axis=1).mean())


def certify(measured, cal, st=0.35, ct=0.75):
    lo, hi = cal.interval(measured)
    lo = max(0, lo)
    if lo >= ct:
        return "changed"
    if hi <= st:
        return "stable"
    return "uncertain"


def run(surfaces, *, noise, alpha, reps, changes, st=0.35, ct=0.75):
    rng = np.random.default_rng(0)
    resid = [measure(s, s.copy(), rng, noise) for s in surfaces for _ in range(reps)]
    cal = ErrorCalibrator.fit(resid, alpha=alpha)
    curve = []
    for ch in changes:
        n = cc = sc = 0
        for s in surfaces:
            t1 = displace(s, ch)
            for _ in range(reps):
                lab = certify(measure(s, t1, rng, noise), cal, st, ct)
                n += 1; cc += lab == "changed"; sc += lab == "stable"
        rate = cc / n
        curve.append({"change_mm": ch, "changed_rate": rate, "stable_rate": sc / n,
                      "recall": rate if ch >= ct else None, "fpr": rate if ch == 0 else None})
    return {"radius_mm": cal.radius_mm, "curve": curve}


def main():
    loader = Poseidon3DLoader(DATA)
    recs = [r for r in loader.records() if r.mesh_path and Path(r.mesh_path).exists()][:8]
    surfaces = [load_poseidon3d_points(r, n_points=2000, seed=0) for r in recs]
    res = {"dataset": "poseidon3d", "n_meshes": len(surfaces), "ablations": {}}
    changes = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]

    print("[main] recon noise 0.05mm (IOS), alpha 0.1", flush=True)
    res["main"] = run(surfaces, noise=0.05, alpha=0.1, reps=12, changes=changes)
    for r in res["main"]["curve"]:
        print(f"  change {r['change_mm']}mm: changed={r['changed_rate']:.2f} stable={r['stable_rate']:.2f}", flush=True)

    print("[noise] reconstruction-noise ablation", flush=True)
    for nz in [0.03, 0.05, 0.10, 0.20]:
        res["ablations"][f"noise_{nz}"] = run(surfaces, noise=nz, alpha=0.1, reps=12, changes=changes)
        c = res["ablations"][f"noise_{nz}"]
        rec1 = next(x["changed_rate"] for x in c["curve"] if x["change_mm"] == 1.0)
        fpr = next(x["changed_rate"] for x in c["curve"] if x["change_mm"] == 0.0)
        print(f"  noise {nz}mm: radius={c['radius_mm']:.3f} recall@1.0mm={rec1:.2f} fpr={fpr:.3f}", flush=True)

    print("[alpha] FPR vs alpha (noise 0.05)", flush=True)
    for a in [0.05, 0.1, 0.2]:
        res["ablations"][f"alpha_{a}"] = run(surfaces, noise=0.05, alpha=a, reps=12, changes=changes)
        fpr = next(x["changed_rate"] for x in res["ablations"][f"alpha_{a}"]["curve"] if x["change_mm"] == 0.0)
        print(f"  alpha {a}: fpr={fpr:.3f} (<= {a}? {fpr <= a})", flush=True)

    OUT.write_text(json.dumps(res, indent=1))
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
