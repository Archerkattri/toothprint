#!/usr/bin/env python3
"""Aggregate the thorough/ablation evaluation results into plots + a summary."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent / "results"


def load(name):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _rec1(block):
    return next(x["changed_rate"] for x in block["curve"] if x["change_mm"] == 1.0)


def main():
    id3d, id2d = load("id3d"), load("id2d")
    change, surface = load("change"), load("surface")
    reg_gt, reg_det = load("change_registration_gt"), load("change_registration_detector")
    reg_yolo = load("change_registration_yolo")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1 — change recall vs magnitude
    if change:
        c = change["magnitude_curve"]
        axes[0, 0].plot([x["shift_px"] for x in c], [x["recall"] for x in c], "-o", color="#1f77b4")
        axes[0, 0].plot([x["shift_px"] for x in c], [x["fpr"] for x in c], "-s", color="#d62728", label="FPR")
        axes[0, 0].set_title("Change: recall vs magnitude (low noise)")
        axes[0, 0].set_xlabel("crestal change (px)"); axes[0, 0].set_ylabel("rate"); axes[0, 0].grid(alpha=.3); axes[0, 0].legend()

    # 2 — change recall vs ACQUISITION NOISE (the limitation)
    if change:
        ns = [1, 3, 5, 8]; rec = [change["ablations"][f"noise_{n}"]["recall"] for n in ns]
        fpr = [change["ablations"][f"noise_{n}"]["fpr"] for n in ns]
        axes[0, 1].plot(ns, rec, "-o", color="#1f77b4", label="recall @ 2mm change")
        axes[0, 1].plot(ns, fpr, "-s", color="#d62728", label="FPR")
        axes[0, 1].set_title("Change: recall collapses with acquisition noise")
        axes[0, 1].set_xlabel("acquisition noise (px)"); axes[0, 1].set_ylabel("rate"); axes[0, 1].grid(alpha=.3); axes[0, 1].legend()

    # 3 — surface recall vs reconstruction noise: de-biasing extends the range
    if surface:
        nz = [0.03, 0.05, 0.10, 0.20, 0.40, 0.84]
        deb = [_rec1(surface["ablations"][f"noise_{n}"]) for n in nz]
        raw = [_rec1(surface["baseline_raw"][f"noise_{n}"]) for n in nz]
        axes[0, 2].plot(nz, deb, "-o", color="#2ca02c", label="de-biased")
        axes[0, 2].plot(nz, raw, "-s", color="#d62728", label="raw mean-norm")
        axes[0, 2].axvline(0.84, color="#e6a93f", ls="--", label="GS recon 0.84mm")
        axes[0, 2].set_title("Surface: de-biasing extends usable noise")
        axes[0, 2].set_xlabel("reconstruction noise (mm)"); axes[0, 2].set_ylabel("recall @ 1mm"); axes[0, 2].grid(alpha=.3); axes[0, 2].legend()

    # 4 — 3D identity robustness to tooth loss
    if id3d:
        keeps = [1.0, 0.75, 0.5, 0.3, 0.2]
        def r1(k):
            if k == 0.75: return id3d["ablations"].get("keep_0.75", id3d.get("main", {})).get("rank1") or id3d["main"]["rank1"]
            return id3d["ablations"].get(f"keep_{k}", {}).get("rank1")
        vals = [(id3d["main"]["rank1"] if k == 0.75 else id3d["ablations"].get(f"keep_{k}", {}).get("rank1")) for k in keeps]
        axes[1, 0].plot(keeps, vals, "-o", color="#9467bd")
        axes[1, 0].set_title("3D identity: Rank-1 vs tooth coverage")
        axes[1, 0].set_xlabel("fraction of arch present"); axes[1, 0].set_ylabel("Rank-1"); axes[1, 0].set_ylim(0, 1.05); axes[1, 0].grid(alpha=.3); axes[1, 0].invert_xaxis()

    # 5 — 3D identity robustness to scan noise
    if id3d:
        ns = [0.0, 0.1, 0.2, 0.4]
        vals = [id3d["ablations"].get(f"noise_{n}", {}).get("rank1") for n in ns]
        axes[1, 1].plot(ns, vals, "-o", color="#9467bd")
        axes[1, 1].set_title("3D identity: Rank-1 vs scan noise")
        axes[1, 1].set_xlabel("scan noise (mm)"); axes[1, 1].set_ylabel("Rank-1"); axes[1, 1].set_ylim(0, 1.05); axes[1, 1].grid(alpha=.3)

    # 6 — 2D identity robustness to jitter
    if id2d:
        js = [0.0, 6.0, 12.0, 20.0]
        vals = [id2d["ablations"].get(f"jitter_{j}", {}).get("rank1") for j in js]
        axes[1, 2].plot(js, vals, "-o", color="#8c564b")
        axes[1, 2].set_title("2D identity: Rank-1 vs landmark jitter")
        axes[1, 2].set_xlabel("jitter (px)"); axes[1, 2].set_ylabel("Rank-1"); axes[1, 2].set_ylim(0, 1.05); axes[1, 2].grid(alpha=.3)

    fig.suptitle("ToothPrint — ablation evaluation across all three mechanisms (real data + synthetic perturbations)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = R.parent / "ablation_summary.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print("wrote", out)

    # console summary
    print("\n================ SUMMARY ================")
    if id3d:
        m = id3d["main"]
        print(f"3D identity (N={m['n']}): Rank-1={m['rank1']:.3f} Rank-5={m['rank5']:.3f} "
              f"AUC={m['auc']:.3f} EER={m['eer']:.3f} d'={m['dprime']:.1f} fail_rate={m.get('registration_fail_rate',0):.2f}")
    if id2d:
        m = id2d["main"]
        print(f"2D identity (N={m['n']}): Rank-1={m['rank1']:.3f} AUC={m['auc']:.3f} EER={m['eer']:.3f} d'={m['dprime']:.2f}")
    if change:
        n = change["ablations"]
        print(f"Change cert: FPR<=alpha at all alpha? "
              f"{all(n[f'alpha_{a}']['fpr'] <= a for a in [0.05,0.1,0.2])}")
        print(f"  recall @2mm change vs noise: " + ", ".join(f"{nz}px={n[f'noise_{nz}']['recall']:.2f}" for nz in [1,3,5,8]))
    if surface:
        a = surface["ablations"]
        print(f"Surface recall@1mm (de-biased) vs recon noise: " +
              ", ".join(f"{nz}mm={_rec1(a[f'noise_{nz}']):.2f}" for nz in [0.05,0.2,0.4,0.84]))
        if "correlated" in surface:
            c = surface["correlated"]
            print(f"  correlated-noise caveat (0.2mm): " +
                  ", ".join(f"corr{cc}={_rec1(c[f'corr_{cc}']):.2f}" for cc in [0.0,0.5,1.0]))
    if reg_gt:
        s = reg_gt["sweep"]; fpr0 = next((x["fpr"] for x in s if x["change_px"] == 0), None)
        big = next(x for x in reversed(s) if x["change_px"] >= 8)
        print(f"Change measurement (GT loc): recall@{big['change_px']:.0f}px={big['certified_change']:.2f}, "
              f"stable-FPR={fpr0:.3f} (conformal-bounded by alpha={reg_gt['alpha']})")
    if reg_det:
        s = reg_det["sweep"]; fpr0 = next((x["fpr"] for x in s if x["change_px"] == 0), None)
        last = s[-1]
        print(f"Change end-to-end (ViTPose):  recall@{last['change_px']:.0f}px={last['certified_change']:.2f}, "
              f"stable-FPR={fpr0:.3f} (detector-localization-limited)")
    if reg_yolo:
        s = reg_yolo["sweep"]; fpr0 = next((x["fpr"] for x in s if x["change_px"] == 0), None)
        last = s[-1]
        print(f"Change end-to-end (YOLO26):   recall@{last['change_px']:.0f}px={last['certified_change']:.2f}, "
              f"stable-FPR={fpr0:.3f} (median 18px localization vs ViTPose 38px)")


if __name__ == "__main__":
    main()
