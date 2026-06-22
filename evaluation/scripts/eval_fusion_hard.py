#!/usr/bin/env python3
"""Prove the multimodal fusion gain (#4) in the HARD regime.

At full quality every modality saturates (IOS Rank-1 1.0), which hides fusion's value. Here
each of the three real biometrics — IOS crowns, CBCT bone, CBCT dental-work — is degraded
*independently* with heavy noise + partial overlap, so none is perfect. Because the
degradations are independent across modalities, score-level fusion averages them out and
should BEAT the best single modality — the actual reason multimodal fusion exists. On the
real paired CBCT+IOS set. Small N for a fast run. Writes fusion_hard.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_multimodal import DATA, ios_cloud, matrix, metrics, norm, zscore
from eval_multimodal_full import cbct_clouds

OUT = Path(__file__).resolve().parents[1] / "results" / "fusion_hard.json"
NPAT, JITTER, KEEP = 36, 0.03, 0.8        # sweet-spot degradation: singles ~0.7, room for fusion to win


def hard_aug(p, rng):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax); a = rng.uniform(-0.5, 0.5)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    q = p @ (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)).T
    c = q.mean(0); n = rng.normal(size=3); n /= np.linalg.norm(n)
    q = q[(q - c) @ n >= np.quantile((q - c) @ n, 1 - KEEP)]                  # partial overlap
    return norm(q + rng.normal(0, JITTER, q.shape))                          # heavy independent noise


def main():
    pids = sorted(p.name for p in DATA.iterdir() if p.is_dir())[:NPAT]
    g = {"ios": [], "bone": [], "work": []}; q = {"ios": [], "bone": [], "work": []}; kept = []
    for pid in pids:
        ic = ios_cloud(pid); bn, wk = cbct_clouds(pid, np.random.default_rng(7))
        if ic is None or bn is None or wk is None:
            continue
        k = len(kept) + 1; kept.append(pid)
        for nm, cl, seed in (("ios", ic, 100), ("bone", bn, 200), ("work", wk, 300)):
            g[nm].append(cl); q[nm].append(hard_aug(cl, np.random.default_rng(seed + k)))   # independent degradation
    print(f"hard regime (jitter {JITTER}, keep {KEEP}), {len(kept)} real patients", flush=True)

    M = {nm: matrix(g[nm], q[nm]) for nm in ("ios", "bone", "work")}
    fuse_ib = zscore(M["ios"]) + zscore(M["bone"])                    # two independent scanners (IOS + CBCT)
    fuse_all = zscore(M["ios"]) + zscore(M["bone"]) + zscore(M["work"])
    r = {"ios_crowns": metrics(M["ios"]), "cbct_bone": metrics(M["bone"]),
         "dental_work": metrics(M["work"]),
         "fuse_ios+bone": metrics(fuse_ib), "fuse_all_three": metrics(fuse_all)}
    singles = ("ios_crowns", "cbct_bone", "dental_work")
    best_r1 = max(r[s]["rank1"] for s in singles)
    best_auc = max(r[s]["auc"] for s in singles)
    res = {"n_patients": len(kept), "regime": f"hard (jitter {JITTER}, keep {KEEP})",
           "best_single_rank1": round(best_r1, 3), "best_single_auc": round(best_auc, 3), **r,
           "iosbone_rank1_gain": round(r["fuse_ios+bone"]["rank1"] - best_r1, 3),
           "iosbone_auc_gain": round(r["fuse_ios+bone"]["auc"] - best_auc, 3),
           "all3_rank1_gain": round(r["fuse_all_three"]["rank1"] - best_r1, 3),
           "all3_auc_gain": round(r["fuse_all_three"]["auc"] - best_auc, 3)}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    for k in ("ios_crowns", "cbct_bone", "dental_work", "fuse_ios+bone", "fuse_all_three"):
        print(f"  {k:16s} Rank-1 {res[k]['rank1']:.3f}  AUC {res[k]['auc']:.3f}", flush=True)
    print(f"  best single: Rank-1 {best_r1:.3f}  AUC {best_auc:.3f}", flush=True)
    print(f"  IOS+bone (independent) gain:  Rank-1 {res['iosbone_rank1_gain']:+.3f}  AUC {res['iosbone_auc_gain']:+.3f}", flush=True)
    print(f"  all-three        gain:  Rank-1 {res['all3_rank1_gain']:+.3f}  AUC {res['all3_auc_gain']:+.3f}", flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
