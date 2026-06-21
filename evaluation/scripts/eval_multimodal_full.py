#!/usr/bin/env python3
"""Powered-up multimodal + dental-work fusion (#4/#5) at scale.

With ~20 patients crown geometry saturated at Rank-1 1.0, hiding any fusion gain. Re-run on
the larger downloaded pool with THREE real biometrics per patient — IOS crown surface, CBCT
bone/root geometry, and the CBCT dental-work (restoration) pattern — and report every single
modality plus their score-level fusions, so the fusion gain is visible where one modality no
longer saturates. Each CBCT is read once. Writes multimodal_full.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_multimodal import DATA, N, augment, ios_cloud, matrix, metrics, norm, zscore

OUT = Path(__file__).resolve().parents[1] / "results" / "multimodal_full.json"
NPAT = 55


def cbct_clouds(pid, rng):
    a = nib.load(str(DATA / pid / f"{pid}_cbct" / f"{pid}_cbct.nii.gz")).get_fdata()

    def dn(idx):
        if len(idx) < 500:
            return None
        if len(idx) > N:
            idx = idx[rng.choice(len(idx), N, replace=False)]
        return norm(idx.astype(np.float32))
    return dn(np.argwhere(a > a.mean() + a.std())), dn(np.argwhere(a > 2500))   # bone, dental-work


def main():
    pids = sorted(p.name for p in DATA.iterdir() if p.is_dir())[:NPAT]
    ios_g, ios_q, bn_g, bn_q, wk_g, wk_q, kept = [], [], [], [], [], [], []
    for pid in pids:
        ic = ios_cloud(pid); bn, wk = cbct_clouds(pid, np.random.default_rng(7))
        if ic is None or bn is None or wk is None:
            continue
        kept.append(pid)
        ios_g.append(ic); bn_g.append(bn); wk_g.append(wk)
        ios_q.append(augment(ic, np.random.default_rng(100 + len(kept))))
        bn_q.append(augment(bn, np.random.default_rng(200 + len(kept))))
        wk_q.append(augment(wk, np.random.default_rng(300 + len(kept))))
    print(f"real paired patients: {len(kept)}", flush=True)

    IOS = matrix(ios_g, ios_q); BONE = matrix(bn_g, bn_q); WORK = matrix(wk_g, wk_q)
    M = {"ios_crowns": IOS, "cbct_bone": BONE, "dental_work": WORK,
         "fuse_crowns+work": zscore(IOS) + zscore(WORK),
         "fuse_all_three": zscore(IOS) + zscore(BONE) + zscore(WORK)}
    res = {"n_patients": len(kept), **{k: metrics(v) for k, v in M.items()}}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    best_single = max(res["ios_crowns"]["rank1"], res["cbct_bone"]["rank1"], res["dental_work"]["rank1"])
    for k in M:
        print(f"  {k:18s} Rank-1 {res[k]['rank1']:.3f}  AUC {res[k]['auc']:.3f}", flush=True)
    print(f"  best single modality {best_single:.3f}  ->  fusion {res['fuse_all_three']['rank1']:.3f}", flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
