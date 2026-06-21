#!/usr/bin/env python3
"""Thorough change-certificate evaluation with ablations (DenPAR, oracle landmarks).

Validates the conformal change certificate: the false-progression rate must stay
<= alpha (the finite-sample guarantee), recall must rise with change magnitude.
Ablates over alpha, acquisition noise, and the decision threshold tau. Change is
SYNTHETIC (an annotation-level crestal shift) on single-timepoint data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from toothprint.bench.benchmark.pipeline import evaluate_pairs
from toothprint.bench.certificate.conformal import AsymmetricConformalInterval
from toothprint.bench.data.denpar_adapter import RealDenparAdapter
from toothprint.bench.data.pair_builder import PairBuilderConfig, build_pairs
from toothprint.bench.eval.metrics import summarize_decisions
from toothprint.bench.score.periodontal import scalar_change_score

OUT = Path("/home/krishi/personal-projects/toothprint/evaluation/results/change.json")
DATA = "data/denpar/extracted/Dataset"


def run(cal, test, *, alpha, tau, shift, noise, seed=42):
    cfg = PairBuilderConfig(acq_noise_std=noise, crestal_shift_px=shift, seed=seed)
    cp = build_pairs(cal, cfg)
    pred = [scalar_change_score(p.baseline, p.followup) for p in cp]
    obs = [p.true_change for p in cp]
    conf = AsymmetricConformalInterval.fit(pred, obs, alpha=alpha)
    rows = evaluate_pairs(build_pairs(test, cfg), tau=tau, conformal=conf)
    s = summarize_decisions(rows)
    return {"recall": s.true_change_recall, "fpr": s.false_progression_rate,
            "stable_cert": s.stable_certification_rate, "uncertain": s.uncertain_rate}


def main():
    adapter = RealDenparAdapter(DATA)
    cal = list(adapter.records("train")); test = list(adapter.records("test"))
    res = {"dataset": "denpar", "n_cal": len(cal), "n_test": len(test), "ablations": {}}

    print("[curve] recall/FPR vs change magnitude (alpha 0.1, tau 8, noise 3)", flush=True)
    res["magnitude_curve"] = [{"shift_px": s, **run(cal, test, alpha=0.1, tau=8, shift=s, noise=3)}
                              for s in [0, 4, 8, 12, 16, 20, 28, 40]]
    for r in res["magnitude_curve"]:
        print(f"  shift {r['shift_px']:>2}px: recall={r['recall']:.3f} fpr={r['fpr']:.3f}", flush=True)

    print("[alpha] does FPR stay <= alpha?  (shift 20, tau 8, noise 3)", flush=True)
    for a in [0.05, 0.1, 0.2]:
        res["ablations"][f"alpha_{a}"] = run(cal, test, alpha=a, tau=8, shift=20, noise=3)
        m = res["ablations"][f"alpha_{a}"]
        print(f"  alpha {a}: fpr={m['fpr']:.3f} (<= {a}? {m['fpr'] <= a})  recall={m['recall']:.3f}", flush=True)

    print("[noise] acquisition noise robustness (alpha 0.1, tau 8, shift 20)", flush=True)
    for n in [1, 3, 5, 8]:
        res["ablations"][f"noise_{n}"] = run(cal, test, alpha=0.1, tau=8, shift=20, noise=n)
        m = res["ablations"][f"noise_{n}"]
        print(f"  noise {n}px: recall={m['recall']:.3f} fpr={m['fpr']:.3f}", flush=True)

    print("[tau] decision-threshold sweep (alpha 0.1, shift 20, noise 3)", flush=True)
    for t in [4, 8, 12, 16]:
        res["ablations"][f"tau_{t}"] = run(cal, test, alpha=0.1, tau=t, shift=20, noise=3)
        m = res["ablations"][f"tau_{t}"]
        print(f"  tau {t}px: recall={m['recall']:.3f} fpr={m['fpr']:.3f} stable_cert={m['stable_cert']:.3f}", flush=True)

    OUT.write_text(json.dumps(res, indent=1))
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
