#!/usr/bin/env python3
"""Change-certificate evidence figure from the real-data registration sweeps.

Left  — recall vs change magnitude for both localizations (GT measurement ceiling
        vs the fully-automatic ViTPose pipeline) at a fixed clinical threshold.
Right — the measurement's recall/false-progression frontier over tau, showing the
        operating point where false-progression reaches a true 0 while recall
        stays near-perfect.

Reads the per-tooth measurements saved in evaluation/results/*.json so the figure
is reproducible from committed data (no re-run, no GPU).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[1] / "results"


def load(name):
    return json.loads((R / f"{name}.json").read_text())


def progressed_rate(rows, key, q_lo, tau):
    """Fraction certified 'progressed' = measured - q_lo > tau."""
    vals = [r[key] for r in rows if key in r]
    return float(np.mean([(m - q_lo) > tau for m in vals])) if vals else float("nan")


def main():
    gt = load("change_registration_gt")
    det = load("change_registration_detector")
    yolo = load("change_registration_yolo")
    mags = sorted({float(k) for r in gt["test_rows"] for k in r})
    changes = [m for m in mags if m > 0]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    tau = 0.5  # sensitive sub-mm clinical threshold (px)

    # ---- Left: recall vs magnitude, all localizations ------------------------
    for d, color, label in [(gt, "#1f77b4", "GT localization (measurement ceiling)"),
                            (yolo, "#11505f", "YOLO26-pose (fully-automatic, 18px)"),
                            (det, "#d6791f", "ViTPose (fully-automatic, 38px)")]:
        if not d:
            continue
        rec = [progressed_rate(d["test_rows"], str(m), d["q_lo"], tau) for m in changes]
        axL.plot(changes, rec, "-o", color=color, lw=2.2, ms=6, label=label)
    axL.axhline(1.0, color="#999", ls=":", lw=1)
    axL.set_title(f"Change recall vs magnitude  (tau={tau:g}px)", fontsize=12)
    axL.set_xlabel("true crestal change (px)"); axL.set_ylabel("recall (certified-progressed rate)")
    axL.set_ylim(0, 1.05); axL.grid(alpha=.3); axL.legend(loc="lower right", fontsize=9)
    axL.annotate("near-perfect: the differential\nregistration measurement",
                 xy=(changes[2], 0.99), xytext=(changes[1], 0.62), fontsize=9, color="#1f77b4",
                 arrowprops=dict(arrowstyle="->", color="#1f77b4"))
    axL.annotate("a precise detector (YOLO26-pose)\ncloses most of the gap",
                 xy=(changes[-2], 0.89), xytext=(changes[1], 0.34), fontsize=9, color="#11505f",
                 arrowprops=dict(arrowstyle="->", color="#11505f"))

    # ---- Right: measurement recall / FPR frontier over tau --------------------
    taus = np.linspace(0.2, 3.0, 29)
    rec8 = [progressed_rate(gt["test_rows"], "8.0", gt["q_lo"], t) for t in taus]
    fprt = [progressed_rate(gt["test_rows"], "0.0", gt["q_lo"], t) for t in taus]
    axR.plot(taus, rec8, "-", color="#1f77b4", lw=2.4, label="recall @ 8px change")
    axR.plot(taus, fprt, "-", color="#d62728", lw=2.4, label="false-progression (stable pairs)")
    # mark the true-0-FPR operating point
    t0 = next((t for t, f in zip(taus, fprt) if f <= 1e-9), None)
    if t0 is not None:
        r0 = progressed_rate(gt["test_rows"], "8.0", gt["q_lo"], t0)
        axR.axvline(t0, color="#2ca02c", ls="--", lw=1.5)
        axR.annotate(f"0% false-progression\nat recall {r0:.2f}  (tau={t0:.1f}px)",
                     xy=(t0, r0), xytext=(t0 + 0.15, 0.55), fontsize=9, color="#2ca02c",
                     arrowprops=dict(arrowstyle="->", color="#2ca02c"))
    axR.set_title("Measurement: false-progression is conformally bounded", fontsize=12)
    axR.set_xlabel("clinical threshold tau (px)"); axR.set_ylabel("rate")
    axR.set_ylim(0, 1.05); axR.grid(alpha=.3); axR.legend(loc="center left", fontsize=9)

    fig.suptitle("ToothPrint — bone-level change certificate on real DenPAR radiographs",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for out in [R.parents[1] / "web" / "assets" / "change_certificate.png",
                R.parents[1] / "docs" / "change_certificate_v2.png"]:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
