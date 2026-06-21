#!/usr/bin/env python3
"""Thorough 2D radiograph dental-identification evaluation with ablations.

Closed-set identification over real DenPAR landmark constellations. Genuine
queries are acquisition-perturbed (similarity reposition + magnification +
landmark jitter), since the dataset is single-timepoint. Ablates over jitter,
magnification, minimum tooth count, and gallery size.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from toothprint.bench.data.denpar_adapter import RealDenparAdapter

OUT = Path("/home/krishi/personal-projects/toothprint/evaluation/results/id2d.json")
DATA = "data/denpar/extracted/Dataset"


def constellation(ann):
    pts = [[float(p[0]), float(p[1])] for t in ann.get("teeth", [])
           for fld in ("cej", "crest_line", "apex") for p in (t.get(fld) or [])]
    return np.asarray(pts) if pts else None


def _norm(c):
    c = c - c.mean(0); r = np.sqrt((c ** 2).sum(1).mean()); return c / (r or 1), (r or 1)


def icp2d(q, g, iters=30):
    s, _ = _norm(q); d, dr = _norm(g); prev = np.inf; rms = np.inf
    for _ in range(iters):
        d2 = ((s[:, None] - d[None]) ** 2).sum(2); nn = d2.argmin(1)
        rms = np.sqrt(d2[np.arange(len(s)), nn].mean())
        sc = s.mean(0); dc = d[nn].mean(0); H = (s - sc).T @ (d[nn] - dc)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        s = (s - sc) @ R.T + dc
        if abs(prev - rms) < 1e-6:
            break
        prev = rms
    return rms * dr


def load_gallery(min_teeth, cap):
    gal, labs = [], []
    for rec in RealDenparAdapter(DATA).records("test"):
        teeth = [t for t in rec.annotation_dict.get("teeth", []) if t.get("cej") and t.get("crest_line")]
        if len(teeth) < min_teeth:
            continue
        c = constellation({"teeth": teeth})
        if c is None:
            continue
        gal.append(c); labs.append(rec.image_id)
        if len(gal) >= cap:
            break
    return gal, labs


def evaluate(gal, labs, mag, jitter, seed=0):
    rng = np.random.default_rng(seed)
    n = len(gal)
    M = np.zeros((n, n))
    for i, c in enumerate(gal):
        a = rng.uniform(-0.15, 0.15); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        q = (1 + rng.uniform(-mag, mag)) * c @ R.T + rng.uniform(-20, 20, 2) + rng.normal(0, jitter, c.shape)
        for j, g in enumerate(gal):
            M[i, j] = icp2d(q, g)
    gen, imp, r1, r5 = [], [], 0, 0
    for i in range(n):
        order = np.argsort(M[i])
        r1 += labs[order[0]] == labs[i]
        r5 += labs[i] in [labs[k] for k in order[:5]]
        for j in range(n):
            (gen if i == j else imp).append(M[i, j])
    gen, imp = np.array(gen), np.array(imp)
    if gen.size == 0 or imp.size == 0:
        return {"n": n, "rank1": r1 / n if n else float("nan"), "rank5": r5 / n if n else float("nan"),
                "auc": float("nan"), "eer": float("nan"), "dprime": float("nan"),
                "genuine_mean": float("nan"), "genuine_max": float("nan"),
                "impostor_mean": float("nan"), "impostor_min": float("nan")}
    auc = float(np.mean(gen[:, None] < imp[None, :]))
    thr = np.unique(np.concatenate([gen, imp]))
    far = np.array([(imp < t).mean() for t in thr]); frr = np.array([(gen > t).mean() for t in thr])
    eer = float(np.min(np.maximum(far, frr)))
    dprime = float(abs(gen.mean() - imp.mean()) / np.sqrt((gen.var() + imp.var()) / 2))
    return {"n": n, "rank1": r1 / n, "rank5": r5 / n, "auc": auc, "eer": eer, "dprime": dprime,
            "genuine_mean": float(gen.mean()), "genuine_max": float(gen.max()),
            "impostor_mean": float(imp.mean()), "impostor_min": float(imp.min())}


def main():
    res = {"dataset": "denpar", "ablations": {}}
    gal, labs = load_gallery(min_teeth=4, cap=120)
    res["n_subjects"] = len(gal)
    print(f"[main] N={len(gal)} subjects, default (mag 0.08, jitter 3px)", flush=True)
    res["main"] = evaluate(gal, labs, mag=0.08, jitter=3.0)
    print(f"  rank1={res['main']['rank1']:.3f} d'={res['main']['dprime']:.2f} eer={res['main']['eer']:.3f}", flush=True)

    for jit in [0.0, 6.0, 12.0, 20.0]:
        res["ablations"][f"jitter_{jit}"] = evaluate(gal, labs, mag=0.08, jitter=jit)
        print(f"  jitter {jit}px: rank1={res['ablations'][f'jitter_{jit}']['rank1']:.3f}", flush=True)
    for mag in [0.0, 0.15, 0.3, 0.5]:
        res["ablations"][f"mag_{mag}"] = evaluate(gal, labs, mag=mag, jitter=3.0)
        print(f"  magnification {mag}: rank1={res['ablations'][f'mag_{mag}']['rank1']:.3f}", flush=True)
    for mt in [3, 6, 8]:
        g, l = load_gallery(min_teeth=mt, cap=120)
        res["ablations"][f"minteeth_{mt}"] = evaluate(g, l, mag=0.08, jitter=3.0)
        print(f"  min-teeth {mt} (N={len(g)}): rank1={res['ablations'][f'minteeth_{mt}']['rank1']:.3f}", flush=True)

    OUT.write_text(json.dumps(res, indent=1))
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
