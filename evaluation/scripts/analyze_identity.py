#!/usr/bin/env python3
"""Full identity analysis on the saved N=200 score matrix — the metrics that make a
SOTA-credible claim, computed post-hoc (no re-registration):

  - standard biometric metrics: Rank-1/5, EER, AUC (+ bootstrap 95% CI), d'
  - CMC curve, DET/ROC
  - **conformal bounded false-match-rate** (the novel claim): split-conformal
    calibration of an accept threshold so empirical FMR <= alpha in finite samples
  - **open-set** identification: hold identities out of the gallery, report
    DIR(rank-1) vs FPIR and FNIR @ FPIR = 1%

Reads outputs/exp/{core_p2p.npy, gen_p2s.npy, imp_p2s.npy}; writes
outputs/exp/identity_analysis.json and docs/identity_metrics.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parents[1] / "results"  # toothprint/evaluation/results
RNG = np.random.default_rng(0)


def split_scores(M):
    n = M.shape[0]
    gen = np.array([M[i, i] for i in range(n)])
    imp = np.array([M[i, j] for i in range(n) for j in range(n) if i != j])
    return gen, imp


def basic_metrics(M):
    n = M.shape[0]
    gen, imp = split_scores(M)
    rank1 = float(np.mean([np.argmin(M[i]) == i for i in range(n)]))
    rank5 = float(np.mean([i in np.argsort(M[i])[:5] for i in range(n)]))
    auc = float(np.mean(gen[:, None] < imp[None, :]))
    thr = np.unique(np.concatenate([gen, imp]))
    far = np.array([(imp < t).mean() for t in thr]); frr = np.array([(gen > t).mean() for t in thr])
    eer = float(np.min(np.maximum(far, frr)))
    dprime = float(abs(gen.mean() - imp.mean()) / np.sqrt((gen.var() + imp.var()) / 2 + 1e-12))
    return dict(n=n, rank1=rank1, rank5=rank5, eer=eer, auc=auc, dprime=dprime,
                gen_mean=float(gen.mean()), gen_median=float(np.median(gen)), gen_max=float(gen.max()),
                imp_mean=float(imp.mean()), imp_min=float(imp.min()), no_overlap=bool(gen.max() < imp.min()))


def bootstrap_auc_ci(M, B=2000):
    """95% CI for AUC by resampling SUBJECTS (the unit of independence)."""
    n = M.shape[0]; aucs = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, n, n)
        sub = M[np.ix_(idx, idx)]
        gen = np.array([sub[i, i] for i in range(n)])
        imp = np.array([sub[i, j] for i in range(n) for j in range(n) if i != j])
        aucs[b] = np.mean(gen[:, None] < imp[None, :])
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def cmc(M):
    n = M.shape[0]
    ranks = np.array([int(np.where(np.argsort(M[i]) == i)[0][0]) for i in range(n)])
    return [float((ranks < k).mean()) for k in range(1, 11)]


def conformal_fmr(M, alphas, trials=200):
    """Split-conformal accept threshold with a finite-sample FMR bound.

    On a calibration split, set tau = the order statistic of impostor scores that
    guarantees P(impostor < tau) <= alpha; measure empirical FMR (and the genuine
    accept rate, TAR) on the disjoint test split. Averaged over random splits.
    """
    n = M.shape[0]
    out = {float(a): {"emp_fmr": [], "tar": []} for a in alphas}
    for _ in range(trials):
        perm = RNG.permutation(n); cal, test = perm[:n // 2], perm[n // 2:]
        cal_imp = np.array([M[i, j] for i in cal for j in cal if i != j])
        test_imp = np.array([M[i, j] for i in test for j in test if i != j])
        test_gen = np.array([M[i, i] for i in test])
        for a in alphas:
            k = max(int(np.floor(a * (len(cal_imp) + 1))) - 1, 0)   # conformal order stat
            tau = np.sort(cal_imp)[k]
            out[float(a)]["emp_fmr"].append(float((test_imp < tau).mean()))
            out[float(a)]["tar"].append(float((test_gen < tau).mean()))
    return {a: {"emp_fmr": float(np.mean(v["emp_fmr"])), "emp_fmr_p95": float(np.percentile(v["emp_fmr"], 95)),
                "tar": float(np.mean(v["tar"]))} for a, v in out.items()}


def open_set(M, held_out=40, taus=200, trials=40):
    """DIR(rank-1) vs FPIR by holding identities out of the gallery."""
    n = M.shape[0]
    lo, hi = np.percentile(M, 0.5), np.percentile(M, 30)
    tau_grid = np.linspace(lo, hi, taus)
    dir_curve = np.zeros(taus); fpir_curve = np.zeros(taus)
    for _ in range(trials):
        perm = RNG.permutation(n)
        nonenr, enr = perm[:held_out], perm[held_out:]
        enr_set = list(enr)
        # enrolled queries: best match within gallery columns `enr`
        gscore, gcorrect = [], []
        for i in enr:
            cols = M[i, enr_set]
            j = enr_set[int(np.argmin(cols))]
            gscore.append(float(cols.min())); gcorrect.append(j == i)
        gscore = np.array(gscore); gcorrect = np.array(gcorrect)
        nscore = np.array([float(M[i, enr_set].min()) for i in nonenr])
        for t, tau in enumerate(tau_grid):
            dir_curve[t] += np.mean((gscore < tau) & gcorrect)
            fpir_curve[t] += np.mean(nscore < tau)
    dir_curve /= trials; fpir_curve /= trials
    # FNIR @ FPIR = 1%
    order = np.argsort(fpir_curve)
    fnir_at = float(1 - np.interp(0.01, fpir_curve[order], dir_curve[order]))
    return dict(fpir=fpir_curve.tolist(), dir=dir_curve.tolist(), fnir_at_fpir_1pct=fnir_at)


def main():
    M = np.load(EXP / "core_p2p.npy")
    gen_p2s = np.load(EXP / "gen_p2s.npy"); imp_p2s = np.load(EXP / "imp_p2s.npy")
    bm = basic_metrics(M)
    bm["auc_ci95"] = bootstrap_auc_ci(M)
    cmc_curve = cmc(M)
    alphas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    conf = conformal_fmr(M, alphas)
    os_ = open_set(M)
    fidelity = dict(gen_mean=float(gen_p2s.mean()), gen_median=float(np.median(gen_p2s)),
                    gen_max=float(gen_p2s.max()), imp_mean=float(imp_p2s.mean()), imp_min=float(imp_p2s.min()))
    result = dict(dataset="Poseidon3D N=200", method="PCA-axis init + Generalized-ICP, point-to-point",
                  metrics=bm, cmc=cmc_curve, alignment_fidelity_point_to_surface=fidelity,
                  conformal_fmr={str(k): v for k, v in conf.items()},
                  open_set={"fnir_at_fpir_1pct": os_["fnir_at_fpir_1pct"], "held_out": 40})
    (EXP / "identity_analysis.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"Rank-1 {bm['rank1']:.3f}  Rank-5 {bm['rank5']:.3f}  EER {bm['eer']:.3f}  "
          f"AUC {bm['auc']:.3f} CI{tuple(round(x,3) for x in bm['auc_ci95'])}  d' {bm['dprime']:.2f}")
    print("conformal FMR:  " + "  ".join(f"a={a}:emp={conf[a]['emp_fmr']:.3f}(TAR {conf[a]['tar']:.2f})" for a in alphas))
    print(f"open-set FNIR@FPIR=1%: {os_['fnir_at_fpir_1pct']:.3f}  | p2s genuine {fidelity['gen_mean']:.3f}mm")

    # --- figure ---
    gen, imp = split_scores(M)
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    hi = np.percentile(np.r_[gen, imp], 99.5); bins = np.linspace(0, hi, 60)
    ax[0].hist(imp, bins=bins, color="#d6543c", alpha=0.55, density=True, label=f"impostor ({len(imp):,})")
    ax[0].hist(gen, bins=bins, color="#2ca06b", alpha=0.85, density=True, label=f"genuine ({len(gen)})")
    ax[0].set_title(f"N={bm['n']} · Rank-1 {bm['rank1']:.3f} · EER {bm['eer']:.3f}\nAUC {bm['auc']:.3f} "
                    f"[{bm['auc_ci95'][0]:.3f},{bm['auc_ci95'][1]:.3f}]", fontsize=10)
    ax[0].set_xlabel("match distance (mm)"); ax[0].legend(fontsize=8)
    ax[1].plot(range(1, 11), cmc_curve, "-o", color="#2c6fb0"); ax[1].set_ylim(min(cmc_curve) - 0.02, 1.005)
    ax[1].set_title("CMC curve"); ax[1].set_xlabel("rank"); ax[1].set_ylabel("identification rate")
    a_arr = np.array(alphas); emp = np.array([conf[a]["emp_fmr"] for a in alphas])
    tar = np.array([conf[a]["tar"] for a in alphas])
    ax[2].plot([0, max(alphas)], [0, max(alphas)], ":", color="#888", label="FMR = α bound")
    ax[2].plot(a_arr, emp, "-o", color="#c0392b", label="empirical FMR (test)")
    ax[2].plot(a_arr, tar, "-s", color="#2ca06b", label="genuine accept (TAR)")
    ax[2].set_title("Conformal bounded false-match rate"); ax[2].set_xlabel("target α"); ax[2].legend(fontsize=8)
    ax[3].plot(os_["fpir"], os_["dir"], color="#7b3fb0")
    ax[3].axvline(0.01, ls=":", color="#888"); ax[3].set_xlim(0, 0.3)
    ax[3].set_title(f"Open-set · DIR vs FPIR\nFNIR@FPIR=1%: {os_['fnir_at_fpir_1pct']:.3f}", fontsize=10)
    ax[3].set_xlabel("false-positive identification rate"); ax[3].set_ylabel("detection+identification rate")
    fig.suptitle("Identity on all 200 Poseidon3D subjects — standard metrics, conformal bounded-FMR, and open-set",
                 fontsize=13, y=1.0)
    fig.patch.set_facecolor("white"); fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = Path(__file__).resolve().parents[2] / "docs" / "identity_metrics.png"; fig.savefig(out, dpi=115); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
