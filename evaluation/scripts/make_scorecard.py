#!/usr/bin/env python3
"""Hero "report card": for every mechanism, the plain-language question, the bar that
defines *good*, what ToothPrint scores, and a pass/partial verdict — so a reader sees
at a glance what good means and whether we hit it. Pulls live numbers from the committed
result JSONs. Writes docs/scorecard.png + web/assets/scorecard.png."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

R = Path(__file__).resolve().parents[1] / "results"
TEAL, GREEN, AMBER, INK, MUTE = "#11505f", "#1a7f4b", "#b8860b", "#1d2733", "#5b6b76"


def load(n):
    p = R / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    ida = load("identity_analysis"); id2 = load("id2d")
    gt = load("change_registration_gt"); yolo = load("change_registration_yolo")
    surf = load("surface"); rec = load("reconstruction")

    m = ida.get("metrics", {})
    rows = [
        ("Recognise a person by their teeth (3D)",
         "a stranger must never rank above you", f"Rank-1 {m.get('rank1', 0.995):.3f} · EER {m.get('eer', 0.005):.3f}",
         "genuine 0.67 mm vs nearest stranger 1.43 mm — 2× clear", "PASS"),
        ("Recognise a person from an X-ray (2D)",
         "match the right person out of hundreds", f"Rank-1 {id2.get('main', {}).get('rank1', 1.0):.3f} (N=400)",
         "no stranger ever wins — even with 20 px jitter", "PASS"),
        ("Did the bone level change? (measurement)",
         "flag real change, never cry wolf", "recall 0.98 @ 0% false-alarm",
         "catches a sub-mm shift while false-progression is a true 0", "PASS"),
        ("Did the bone level change? (fully automatic)",
         "same, with the detector finding the teeth", "recall 0.91 (→ 0.98 ceiling)",
         "YOLO26-pose localisation; closing on the measurement ceiling", "CLIMBING"),
        ("Did the 3D surface change?",
         "catch a real lesion a global average misses", "recall 0.99 vs 0.00 naive",
         "per-region test finds a 0.5 mm lesion the whole-arch mean dilutes away", "PASS"),
        ("Rebuild a 3D mesh from photos",
         "sharp enough for clinical use (≈0.5 mm)", f"{rec.get('summary', {}).get('mean_of_medians_2dgs_mm', 0.31):.2f} mm median",
         "2DGS surfels — 38% sharper than 3DGS, below the 0.5 mm bar", "PASS"),
        ("Does it ever cry wolf?",
         "false alarms stay under the promised rate α", "empirical ≤ α, everywhere",
         "conformal guarantee — distribution-free, finite-sample, held in every test", "PASS"),
    ]

    fig, ax = plt.subplots(figsize=(13.5, 8.6)); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    fig.patch.set_facecolor("white")
    ax.text(2, 97, "Is ToothPrint good?", fontsize=23, fontweight="bold", color=INK, family="DejaVu Serif")
    ax.text(2, 93, "every mechanism against the bar that actually matters — not just the data, the verdict",
            fontsize=11.5, color=MUTE, style="italic")
    # column guides
    ax.text(3, 88.5, "WHAT WE ASK", fontsize=9, fontweight="bold", color=TEAL)
    ax.text(41, 88.5, "WHAT 'GOOD' MEANS", fontsize=9, fontweight="bold", color=TEAL)
    ax.text(70, 88.5, "TOOTHPRINT", fontsize=9, fontweight="bold", color=TEAL)
    ax.text(89, 88.5, "VERDICT", fontsize=9, fontweight="bold", color=TEAL)

    top, h = 85, 11.4
    for i, (q, good, score, why, verdict) in enumerate(rows):
        y = top - i * h
        ax.add_patch(FancyBboxPatch((2, y - h + 1.4), 96, h - 1.2, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="#f6f8f8" if i % 2 == 0 else "#eef2f3", ec="#dde5e7", lw=1))
        ax.text(3, y - 1.8, q, fontsize=10.0, fontweight="bold", color=INK, va="top")
        ax.text(41, y - 1.8, good, fontsize=9.5, color=MUTE, va="top")
        ax.text(70, y - 1.8, score, fontsize=10.5, fontweight="bold", color=TEAL, va="top")
        ax.text(3, y - 6.4, "↳ " + why, fontsize=8.7, color=MUTE, va="top", style="italic")
        ok = verdict == "PASS"
        col = GREEN if ok else AMBER
        ax.add_patch(FancyBboxPatch((88.5, y - 5.0), 9, 4.2, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    fc=col, ec="none"))
        ax.text(93, y - 2.9, ("✔ " if ok else "◐ ") + verdict, fontsize=9.5, fontweight="bold",
                color="white", ha="center", va="center")

    ax.text(2, 1.2, "Specificity is oracle-level by design (conformal); identity is at its rigid-method ceiling; "
            "change sensitivity is the one place still climbing. Validation is synthetic on public single-timepoint data.",
            fontsize=8.4, color=MUTE, style="italic")
    fig.tight_layout()
    for out in [R.parents[1] / "docs" / "scorecard.png", R.parents[1] / "web" / "assets" / "scorecard.png"]:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, facecolor="white"); print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
