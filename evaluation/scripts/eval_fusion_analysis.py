#!/usr/bin/env python3
"""Definitive multimodal-fusion analysis (#4): can fusion beat the best single modality here?

Earlier runs showed equal-weight score fusion ties (saturated regime) or *hurts* (when one
modality dominates, its noise dilutes the strong one). This settles it properly:

  * three real biometrics per patient (IOS crowns, CBCT bone, CBCT dental-work),
  * each degraded independently into a hard regime so none saturates,
  * fusion by (a) equal z-sum and (b) per-query QUALITY weighting (down-weight the
    modality that is unsure for that query — the established biometric-fusion method),
  * and the ORACLE BOUND: the fraction of queries whose true match is rank-1 in *any*
    modality — the hard ceiling on what ANY fusion can reach.

If oracle ≈ best-single, the modalities fail on the *same* queries (correlated failure) and
no fusion can help — an honest, rigorous negative, not a tuning artifact. The N×N matrices are
cached so the verdict is reproducible without recomputing GICP. Writes fusion_analysis.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_fusion_hard import hard_aug
from eval_multimodal import DATA, ios_cloud, matrix, metrics, zscore
from eval_multimodal_full import cbct_clouds

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "fusion_analysis.json"
NPAT, JITTER, KEEP = 30, 0.03, 0.8
CACHE = RES / f"fusion_matrices_n{NPAT}_{str(JITTER).replace('.', 'p')}_{str(KEEP).replace('.', 'p')}.npz"


def qweight(M):
    """Per-query confidence = normalized gap between best and 2nd-best gallery distance."""
    s = np.sort(M, axis=1)
    g = (s[:, 1] - s[:, 0]) / (M.std(axis=1) + 1e-9)
    return (g / (g.sum() + 1e-9))[:, None]                       # column vector of row weights


def oracle_rank1(mats):
    """Upper bound on any fusion: true match is argmin in AT LEAST ONE modality."""
    n = len(mats[0])
    return float(np.mean([any(np.argmin(M[i]) == i for M in mats) for i in range(n)]))


def main():
    if CACHE.exists():
        z = np.load(CACHE); IOS, BONE, WORK = z["ios"], z["bone"], z["work"]
        print(f"loaded cached matrices {CACHE.name} (N={len(IOS)})", flush=True)
    else:
        pids = sorted(p.name for p in DATA.iterdir() if p.is_dir())[:NPAT]
        g = {"ios": [], "bone": [], "work": []}; q = {"ios": [], "bone": [], "work": []}; kept = []
        for pid in pids:
            ic = ios_cloud(pid); bn, wk = cbct_clouds(pid, np.random.default_rng(7))
            if ic is None or bn is None or wk is None:
                continue
            k = len(kept) + 1; kept.append(pid)
            for nm, cl, seed in (("ios", ic, 100), ("bone", bn, 200), ("work", wk, 300)):
                g[nm].append(cl); q[nm].append(hard_aug(cl, np.random.default_rng(seed + k)))
        print(f"hard regime (jitter {JITTER}, keep {KEEP}), {len(kept)} real patients", flush=True)
        IOS, BONE, WORK = (matrix(g[nm], q[nm]) for nm in ("ios", "bone", "work"))
        np.savez(CACHE, ios=IOS, bone=BONE, work=WORK)
        print(f"cached matrices -> {CACHE.name}", flush=True)

    Zi, Zb, Zw = zscore(IOS), zscore(BONE), zscore(WORK)
    wi, wb, ww = qweight(IOS), qweight(BONE), qweight(WORK)
    variants = {
        "ios_crowns": IOS, "cbct_bone": BONE, "dental_work": WORK,
        "equal_all_three": Zi + Zb + Zw,
        "equal_ios+work": Zi + Zw,
        "qweighted_all_three": wi * Zi + wb * Zb + ww * Zw,            # established quality-weighted fusion
    }
    r = {k: metrics(v) for k, v in variants.items()}
    best_single = max(r["ios_crowns"]["rank1"], r["cbct_bone"]["rank1"], r["dental_work"]["rank1"])
    oracle = oracle_rank1([IOS, BONE, WORK])
    res = {"n_patients": len(IOS), "regime": f"hard (jitter {JITTER}, keep {KEEP})",
           "best_single_rank1": round(best_single, 3),
           "oracle_rank1_any_modality": round(oracle, 3),
           "oracle_headroom_over_best_single": round(oracle - best_single, 3), **r,
           "best_fusion_rank1": round(max(r["equal_all_three"]["rank1"], r["equal_ios+work"]["rank1"],
                                          r["qweighted_all_three"]["rank1"]), 3)}
    res["fusion_gain_over_best_single"] = round(res["best_fusion_rank1"] - best_single, 3)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    for k in variants:
        print(f"  {k:22s} Rank-1 {r[k]['rank1']:.3f}  AUC {r[k]['auc']:.3f}", flush=True)
    print(f"  best single  {best_single:.3f}   best fusion {res['best_fusion_rank1']:.3f}   "
          f"gain {res['fusion_gain_over_best_single']:+.3f}", flush=True)
    print(f"  ORACLE (any modality right) {oracle:.3f}  -> headroom over best single "
          f"{res['oracle_headroom_over_best_single']:+.3f}", flush=True)
    verdict = ("fusion CAN help (oracle headroom exists)" if oracle - best_single > 0.03
               else "correlated failure: no fusion can beat the best single here")
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
