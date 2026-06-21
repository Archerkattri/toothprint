#!/usr/bin/env python3
"""Dental work as an explicit identifying signal (#5) — the classic forensic identifier.

In forensic odontology a person's *pattern of restorations* (fillings, crowns, implants) is
one of the strongest identifiers — more distinctive than tooth shape, and it doesn't change
between visits. CBCT makes it directly observable: restorations are metal/ceramic, far denser
than enamel or bone (HU > ~2500). We extract each patient's high-density "dental-work cloud"
and match it as a biometric (PCA+GICP on the restoration pattern), then fuse it with the
geometric IOS identity. On the real paired CBCT+IOS set. Writes dentalwork_identity.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_multimodal import DATA, N, augment, ios_cloud, matrix, metrics, norm, zscore

OUT = Path(__file__).resolve().parents[1] / "results" / "dentalwork_identity.json"
HU = 2500          # restoration threshold — metal/ceramic, well above enamel (~1800)


def dentalwork_cloud(pid, rng):
    a = nib.load(str(DATA / pid / f"{pid}_cbct" / f"{pid}_cbct.nii.gz")).get_fdata()
    idx = np.argwhere(a > HU).astype(np.float32)                    # the restorations, in 3D
    if len(idx) < 500:
        return None
    if len(idx) > N:
        idx = idx[rng.choice(len(idx), N, replace=False)]
    return norm(idx)


def main():
    pids = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    dw_g, dw_q, ios_g, ios_q, kept = [], [], [], [], []
    for pid in pids:
        dw = dentalwork_cloud(pid, np.random.default_rng(7)); ic = ios_cloud(pid)
        if dw is None or ic is None:
            continue
        kept.append(pid)
        dw_g.append(dw); ios_g.append(ic)
        dw_q.append(augment(dw, np.random.default_rng(300 + len(kept))))
        ios_q.append(augment(ic, np.random.default_rng(100 + len(kept))))
    print(f"patients with detectable dental work: {len(kept)}/{len(pids)}", flush=True)

    DW = matrix(dw_g, dw_q)                                          # restoration pattern alone
    IOS = matrix(ios_g, ios_q)                                       # crown geometry alone
    F = zscore(DW) + zscore(IOS)                                     # geometry + dental work
    res = {"n_patients": len(kept), "hu_threshold": HU,
           "dental_work": metrics(DW), "ios_geometry": metrics(IOS), "fused": metrics(F)}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    for k in ("dental_work", "ios_geometry", "fused"):
        print(f"  {k:13s} Rank-1 {res[k]['rank1']:.3f}  AUC {res[k]['auc']:.3f}", flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
