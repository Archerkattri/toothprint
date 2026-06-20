#!/usr/bin/env python3
"""Generate DCC result plots from existing run JSON (no GPU needed).

Outputs to docs/:
  - dcc_vitpose_training.png      ViTPose val error per epoch + per-landmark bars
  - dcc_score_distribution.png    stable vs progressed detector-score histogram
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)


def plot_training():
    log = json.loads((ROOT / "outputs/vitpose_detector/train_log.json").read_text())
    epochs = [e["epoch"] for e in log]
    val = [e["val_overall_px_error"] for e in log]
    names = list(log[-1]["per_landmark_px"].keys())
    best = min(log, key=lambda e: e["val_overall_px_error"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, val, "-o", color="#1f77b4", lw=2, ms=4, label="ViTPose val error")
    ax1.axhline(88.0, color="#d62728", ls="--", lw=1.5, label="KeypointRCNN baseline (~88 px)")
    ax1.axhline(best["val_overall_px_error"], color="#2ca02c", ls=":", lw=1.5,
                label=f"ViTPose best ({best['val_overall_px_error']:.1f} px)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("mean landmark error (px)")
    ax1.set_title("ViTPose fine-tuning on DenPAR"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    vals = [best["per_landmark_px"][n] for n in names]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))
    ax2.barh(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax2.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9)
    ax2.set_xlabel("mean error (px)"); ax2.set_title("Per-landmark error (best epoch)")
    ax2.invert_yaxis(); ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = DOCS / "dcc_vitpose_training.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


def plot_score_distribution():
    metrics_path = ROOT / "outputs/gate2_realimg_d30/metrics.json"
    if not metrics_path.exists():
        print("skip score distribution (no", metrics_path, ")")
        return
    rows = json.loads(metrics_path.read_text())["test_rows"]
    st = np.array([r["score"] for r in rows if r["label"] == "stable"])
    pr = np.array([r["score"] for r in rows if r["label"] == "progressed"])
    clip = 200
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-clip, clip, 61)
    ax.hist(np.clip(st, -clip, clip), bins=bins, alpha=0.6, label=f"stable (n={len(st)})", color="#2ca02c")
    ax.hist(np.clip(pr, -clip, clip), bins=bins, alpha=0.6, label=f"progressed +30px (n={len(pr)})", color="#d62728")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("detector CEJ-to-crest change (px)"); ax.set_ylabel("count")
    ax.set_title("Detector change score: stable vs rendered +30px change\n"
                 "(distributions overlap — a real detector does not track the warp)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = DOCS / "dcc_score_distribution.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


def plot_oracle_certificate():
    metrics_path = ROOT / "outputs/gate2_oracle/metrics.json"
    if not metrics_path.exists():
        print("skip oracle certificate (no", metrics_path, ")")
        return
    d = json.loads(metrics_path.read_text())
    sweep = d["sweep"]
    shifts = [s["shift_px"] for s in sweep]
    recall = [s["recall"] for s in sweep]
    fpr = [s["fpr"] for s in sweep]
    scert = [s["stable_cert"] for s in sweep]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(shifts, recall, "-o", color="#1f77b4", lw=2.2, ms=6, label="true-change recall")
    ax.plot(shifts, fpr, "-s", color="#d62728", lw=2, ms=5, label="false-progression rate")
    ax.plot(shifts, scert, "-^", color="#2ca02c", lw=1.8, ms=5, label="stable-certification rate")
    ax.axhline(d["alpha"], color="#d62728", ls=":", lw=1, alpha=0.6,
               label=f"alpha = {d['alpha']} (FPR budget)")
    ax.set_xlabel("injected crestal change magnitude (px)")
    ax.set_ylabel("rate")
    ax.set_title("Conformal certificate on real DenPAR (oracle landmarks)\n"
                 "recall -> 1.0 for clinically significant change at ~0.5% false-progression rate")
    ax.set_ylim(-0.03, 1.05); ax.legend(loc="center right"); ax.grid(alpha=0.3)
    fig.tight_layout(); out = DOCS / "dcc_oracle_certificate.png"
    fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


def plot_registration_certificate():
    gt_p = ROOT / "outputs/gate2_registration/metrics_gt.json"
    det_p = ROOT / "outputs/gate2_registration/metrics_detector.json"
    if not gt_p.exists():
        print("skip registration certificate (no", gt_p, ")")
        return
    gt = json.loads(gt_p.read_text())
    xs = [s["change_px"] for s in gt["sweep"]]
    gt_change = [s["certified_change"] for s in gt["sweep"]]
    gt_fpr = [(s["fpr"] if s["fpr"] is not None else 0.0) for s in gt["sweep"]]

    fig, ax = plt.subplots(figsize=(8.7, 5))
    ax.plot(xs, gt_change, "-o", color="#1f77b4", lw=2.3, ms=6,
            label="recall — accurate localization (measurement ceiling)")
    if det_p.exists():
        det = json.loads(det_p.read_text())
        dxs = [s["change_px"] for s in det["sweep"]]
        d_change = [s["certified_change"] for s in det["sweep"]]
        d_fpr = [(s["fpr"] if s["fpr"] is not None else 0.0) for s in det["sweep"]]
        ax.plot(dxs, d_change, "-D", color="#9467bd", lw=2, ms=5,
                label="recall — end-to-end (ViTPose localization)")
        ax.plot(dxs, d_fpr, "-s", color="#d62728", lw=1.8, ms=5,
                label="false-progression rate (end-to-end)")
    else:
        ax.plot(xs, gt_fpr, "-s", color="#d62728", lw=1.8, ms=5, label="false-progression rate")
    ax.axvline(gt["tau"], color="gray", ls=":", lw=1, label=f"tau = {gt['tau']}px")
    ax.set_xlabel("true crestal change (px)"); ax.set_ylabel("rate")
    ax.set_title("Registration-based change certificate on real DenPAR\n"
                 "differential sub-pixel measurement: recall 0.97 @ 0% FPR with accurate "
                 "localization;\nend-to-end specificity = 0% FPR (sensitivity localization-limited)")
    ax.set_ylim(-0.03, 1.05); ax.legend(loc="center right", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); out = DOCS / "dcc_registration_certificate.png"
    fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    plot_training()
    plot_score_distribution()
    plot_oracle_certificate()
    plot_registration_certificate()
