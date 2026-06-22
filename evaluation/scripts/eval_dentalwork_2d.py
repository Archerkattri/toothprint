#!/usr/bin/env python3
"""#4: does the restoration (dental-work) pattern identify people on 2D radiographs?

#5 reads the restoration cloud from CBCT (HU>2500). 2D radiographs are the common modality and
restorations are radio-opaque there too — but global thresholding on 8-bit periapicals is
over-saturated and grabs bright *anatomy* (~30 blobs/image), not fillings. The principled
extractor is PER-TOOTH LOCAL CONTRAST: within each DenPAR tooth mask, a restoration is the
patch far brighter than that tooth's own median (realistic: 1-4 restoration-teeth/image). We
canonicalise by the tooth-region frame and match restoration constellations (Chamfer) under
synthetic re-acquisition, on restoration-bearing subjects. An honest test of whether the *2D*
restoration pattern is an independent identifier — or too sparse on periapicals. Writes
dentalwork_2d.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths

D = paths.DENPAR
OUT = Path(__file__).resolve().parents[1] / "results" / "dentalwork_2d.json"
MIN_REST = 3


def extract(iid, split):
    img = D / split / "Images" / f"{iid}.jpg"; mfold = D / split / "Masks (Tooth-wise)" / iid
    if not mfold.is_dir():
        return None
    im = np.asarray(Image.open(img).convert("L")).astype(float)
    cents, toothpix = [], []
    for f in sorted(mfold.glob("*.png")):
        tm = np.asarray(Image.open(f).convert("L"))
        if tm.shape != im.shape:
            continue
        tm = tm > 0
        if tm.sum() < 200:
            continue
        toothpix.append(np.argwhere(tm))
        v = im[tm]; med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-6
        rest = tm & (im > med + 4 * mad) & (im > 200)                       # bright relative to its OWN tooth
        if rest.sum() >= 15:
            ys, xs = np.where(rest); cents.append([xs.mean(), ys.mean()])
    if len(cents) < MIN_REST or not toothpix:
        return None
    allp = np.concatenate(toothpix)[:, ::-1]                                 # (x,y)
    return (np.array(cents) - allp.mean(0)) / (allp.std() + 1e-9)            # canonical tooth-region frame


def augment(pts, rng, jitter, drop):
    a = rng.uniform(-0.2, 0.2); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    q = pts @ R.T * rng.uniform(0.9, 1.1) + rng.normal(0, jitter, pts.shape)
    if drop and len(q) > MIN_REST:                                          # a restoration missed in re-acquisition
        q = q[rng.permutation(len(q))[:-1]]
    return q


def chamfer(a, b):
    d = ((a[:, None] - b[None]) ** 2).sum(2)
    return float(d.min(1).mean() + d.min(0).mean())


def rank1_auc(subs, jitter, drop):
    n = len(subs); M = np.zeros((n, n))
    for i in range(n):
        q = augment(subs[i], np.random.default_rng(100 + i), jitter, drop)
        for j in range(n):
            M[i, j] = chamfer(q, subs[j])
    r1 = float(np.mean([np.argmin(M[i]) == i for i in range(n)]))
    gen = np.array([M[i, i] for i in range(n)])
    imp = np.array([M[i, j] for i in range(n) for j in range(n) if i != j])
    return r1, float(np.mean(gen[:, None] < imp[None, :]))


def main():
    subs = []
    for split in ("Training", "Testing", "Validation"):
        idir = D / split / "Images"
        if not idir.is_dir():
            continue
        for p in sorted(idir.glob("*.jpg")):
            c = extract(p.stem, split)
            if c is not None:
                subs.append(c)
    n = len(subs)
    print(f"restoration-bearing subjects (>= {MIN_REST} restorations): {n}", flush=True)
    if n < 5:
        OUT.write_text(json.dumps({"n_subjects": n, "verdict": "too few restoration-bearing subjects"}, indent=1) + "\n")
        return
    regimes = {"easy(jit0.02)": (0.02, False), "hard(jit0.05+drop1)": (0.05, True),
               "harder(jit0.10+drop1)": (0.10, True)}
    abl = {}
    for name, (j, d) in regimes.items():
        r1, auc = rank1_auc(subs, j, d)
        abl[name] = {"rank1": round(r1, 3), "auc": round(auc, 3)}
        print(f"  {name:22s} Rank-1 {r1:.3f}  AUC {auc:.3f}  (chance {1/n:.3f})", flush=True)
    res = {"n_subjects": n, "min_restorations": MIN_REST, "chance_rank1": round(1 / n, 3),
           "extraction": "per-tooth local contrast (bright vs own-tooth median); ~1-4 restorations/image",
           "robustness_ablation": abl,
           "caveat": "single-timepoint; synthetic re-acquisition perturbs positions not which restorations are detected; restoration-bearing subjects only"}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
