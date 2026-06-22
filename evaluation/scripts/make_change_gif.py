#!/usr/bin/env python3
"""Clear 'change certificate in action' GIF.

On a real DenPAR tooth the periodontal **bone crest recedes from the (fixed) CEJ** between
visits; sub-pixel registration measures the recession and the certificate flips to CHANGED once
it clears the clinical threshold. Redesigned for legibility over the old version: the *lost-bone
band* is shaded red, the CEJ is pinned as the fixed reference, baseline vs. current crest are
drawn explicitly, a CEJ->crest ruler grows, and a gauge fills toward the threshold with an
unambiguous verdict pill. Writes docs/change_measurement.gif.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths

from toothprint.bench.data.denpar_adapter import RealDenparAdapter
from toothprint.bench.perturb.image_change import render_crestal_change
from toothprint.change.registration import measure_change
from toothprint.geometry import mean_point

MMPP, TAU = 0.10, 6.0
OUT = Path(__file__).resolve().parents[1].parent / "docs" / "change_measurement.gif"
INK, GREEN, ROSE, AMBER, SKY = "#1f2a37", "#16a34a", "#e11d48", "#d97706", "#0ea5e9"


def pick_tooth():
    for rec in RealDenparAdapter(str(paths.DENPAR)).records("test"):
        img = np.array(Image.open(rec.image_path).convert("RGB"))
        for tooth in rec.annotation_dict.get("teeth", []):
            if tooth.get("cej") and tooth.get("crest_line"):
                cej, crest = np.array(mean_point(tooth["cej"])), np.array(mean_point(tooth["crest_line"]))
                if 70 < np.linalg.norm(crest - cej) < 200:
                    return img, cej, crest, float(np.linalg.norm(crest - cej))
    raise SystemExit("no suitable tooth")


def main():
    img, cej, crest, L = pick_tooth()
    u = (crest - cej) / L
    g0 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ref_c = cej - 0.6 * L * u
    cx, cy = (cej + crest) / 2; r = int(max(L * 1.7, 95))
    x1, y1 = int(max(0, cx - r)), int(max(0, cy - r))
    x2, y2 = int(min(img.shape[1], cx + r)), int(min(img.shape[0], cy + r))
    o = np.array([x1, y1]); cej_l, crest_l = cej - o, crest - o
    perp = np.array([-u[1], u[0]]); w = L * 0.42

    deltas = list(np.linspace(0, 20, 21)); deltas += deltas[::-1]
    fdir = OUT.parent / "_chframes"; fdir.mkdir(parents=True, exist_ok=True)
    for i, delta in enumerate(deltas):
        warp = render_crestal_change(img, cej.tolist(), crest.tolist(), float(delta)) if delta > 0.1 else img
        g1 = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY).astype(np.float32)
        out = measure_change(g0, g1, ref_c, crest, u, half=20, search=70)
        measured = max(0.0, out[0]) if out else 0.0
        mm = measured * MMPP; changed = measured > TAU
        crop = warp[y1:y2, x1:x2]; crest_now = crest_l + u * delta

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [1, 0.95]})
        axL.imshow(crop); axL.axis("off"); axL.set_title("radiograph — later visit", fontsize=11, color=INK)
        axL.add_patch(Polygon([crest_l - perp * w, crest_l + perp * w, crest_now + perp * w, crest_now - perp * w],
                              closed=True, facecolor=ROSE, alpha=0.28, edgecolor="none", zorder=4))   # lost bone
        axL.add_patch(Circle(cej_l, 4.5, color=SKY, zorder=6))
        axL.annotate("CEJ (fixed)", cej_l, textcoords="offset points", xytext=(8, -3), fontsize=9, color="#0369a1", fontweight="bold")
        for c, col in ((crest_l, GREEN), (crest_now, ROSE)):
            a, b = c - perp * w, c + perp * w
            axL.plot([a[0], b[0]], [a[1], b[1]], "-", color=col, lw=2.6, zorder=6)
        axL.annotate("", xy=tuple(crest_now), xytext=tuple(cej_l), arrowprops=dict(arrowstyle="<->", color="#ca8a04", lw=2), zorder=5)
        mid = (cej_l + crest_now) / 2 + perp * w * 1.25
        axL.text(mid[0], mid[1], f"bone level\n{measured:.1f}px ≈ {mm:.2f}mm", fontsize=8.5, color="#7a5a00", ha="center", va="center", fontweight="bold")

        axR.axis("off")
        axR.text(0.5, 0.9, "Bone-level recession since baseline", ha="center", fontsize=12.5, color=INK, fontweight="bold", transform=axR.transAxes)
        axR.text(0.5, 0.73, f"{measured:.1f} px   ≈   {mm:.2f} mm", ha="center", fontsize=22, family="monospace", color=INK, transform=axR.transAxes)
        gx, gy, gw, gh, maxv = 0.1, 0.5, 0.8, 0.08, TAU * 2.2
        axR.add_patch(Rectangle((gx, gy), gw, gh, transform=axR.transAxes, facecolor="#eef2f7", edgecolor="#cbd5e1"))
        axR.add_patch(Rectangle((gx, gy), gw * min(measured / maxv, 1.0), gh, transform=axR.transAxes,
                                facecolor=(ROSE if changed else AMBER), edgecolor="none"))
        tx = gx + gw * (TAU / maxv)
        axR.plot([tx, tx], [gy - 0.025, gy + gh + 0.025], color=INK, lw=1.6, transform=axR.transAxes)
        axR.text(tx, gy + gh + 0.055, f"clinical threshold τ = {TAU:.0f} px", ha="center", fontsize=8.5, color=INK, transform=axR.transAxes)
        col = GREEN if changed else AMBER
        axR.add_patch(Rectangle((0.16, 0.25), 0.68, 0.12, transform=axR.transAxes, facecolor=col, alpha=0.14, edgecolor=col, lw=1.6))
        axR.text(0.5, 0.31, "CHANGED — certified" if changed else "stable · below threshold", ha="center", fontsize=14, color=col, fontweight="bold", transform=axR.transAxes)
        axR.text(0.5, 0.12, "conformal: false-progression bounded by α", ha="center", fontsize=9, color="#94a3b8", transform=axR.transAxes)

        fig.suptitle("Change certificate — the bone crest recedes from the fixed CEJ; sub-pixel registration measures the recession and certifies it past threshold",
                     fontsize=9.5, y=0.985, color=INK)
        fig.patch.set_facecolor("white"); fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(fdir / f"f{i:03d}.png", dpi=92); plt.close(fig)

    pal = fdir / "pal.png"
    subprocess.run(["ffmpeg", "-y", "-framerate", "12", "-i", str(fdir / "f%03d.png"),
                    "-vf", "fps=12,scale=900:-1:flags=lanczos,palettegen=stats_mode=diff", str(pal)], check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-framerate", "12", "-i", str(fdir / "f%03d.png"), "-i", str(pal),
                    "-lavfi", "fps=12,scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3", str(OUT)],
                   check=True, capture_output=True)
    for f in fdir.glob("*.png"):
        f.unlink()
    fdir.rmdir()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
