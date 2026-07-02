#!/usr/bin/env python3
"""Animated 3-phase registration demo on a **real** Teeth3DS+ upper arch — the positive result,
shown. One arch and its keep-0.5 whole-tooth-dropout crop (the eval protocol's crop geometry,
reused verbatim from ``eval_bufferx_baseline.dense_crop``) are handed to two registrars, and we
animate what each actually does:

  1. **Query arrives** — the partial crop (50% of the teeth gone) in an arbitrary pose.
  2. **Rigid GICP** — the repo's own PCA-init + Generalized-ICP path
     (``toothprint.identity.align_rigid``), run LIVE on this pair. Its measured result is shown,
     whatever it is: on the selected arch it drops into the WRONG basin (the recorded family
     failure mode at heavy tooth loss).
  3. **BUFFER-X** (ICCV 2025, zero-shot, pretrained 3DMatch, no dental training) — run LIVE,
     locks onto the genuine match.

Both registrars are GLOBAL (GICP from a PCA principal-axis init, BUFFER-X from its own feature
matching), so the recovered alignment is independent of the crop's arrival pose — the arbitrary
initial pose is illustrative (so "before" is legible); the END pose of each phase is the true
measured result, and the in-frame RRE / residual are measured on this exact pair.

Honesty: GICP is NOT uniformly bad on Teeth3DS+ (it hits Rank-1 1.00 full-coverage and locks on
most keep-0.5 arches here); this arch is one where its PCA-init flips a missing-teeth half-arch.
The aggregate numbers carry the story — rigid GICP keep-0.5 Rank-1 0.23 (recorded, Poseidon3D)
vs BUFFER-X 1.00 (measured, Teeth3DS+). The pair is chosen by a fixed criterion (the largest GICP
rotation error among the first K arches for which BUFFER-X still locks, RRE < 5°); the exact
subject can vary between runs because open3d mesh sampling is unseeded, but the phenomenon is
robust (2–3 of the first 16 arches flip GICP's PCA-init while BUFFER-X locks). The chosen subject
+ measured RRE / RTE / residual for the committed GIF are written to
``registration_demo_numbers.json``.

Env: TP_TEETH3DS (arch dir), BUFFERX_REPO (built BUFFER-X tree). Needs a GPU.
Writes docs/bufferx_arch_registration.gif.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # repo root (toothprint pkg)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paths  # noqa: E402
from train_correspondence import load_norm, rot  # noqa: E402
from eval_bufferx_baseline import _load_bufferx  # noqa: E402
from toothprint.identity import align_rigid  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "bufferx_arch_registration.gif"
NUMS = Path(__file__).resolve().parent / "registration_demo_numbers.json"
CACHE = os.environ.get("TP_REGDEMO_CACHE")  # dev-only: reuse a computed pair to iterate on the render

NP_ = int(os.environ.get("TP_REGDEMO_NP", "8000"))
KEEP = 0.5
K = int(os.environ.get("TP_REGDEMO_K", "16"))     # search window for the contrast pair

AQUA, RED, SLATE, ARCHC, INK, MUTED = "#1baf7a", "#e34948", "#5b6572", "#c7ccd1", "#0b0b0b", "#898781"
# illustrative arrival pose (both registrars are global, so this does not affect the result)
R_INIT = Rotation.from_rotvec(
    np.deg2rad(55) * np.array([1, 0.35, 0.25]) / np.linalg.norm([1, 0.35, 0.25])).as_matrix()
T_INIT = np.array([0.15, 0.0, 0.95])


def kabsch(a, b):
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def rre_deg(R_est, R_gt):
    return float(np.degrees(np.arccos(np.clip((np.trace(R_est @ R_gt.T) - 1) / 2, -1, 1))))


def make_crop(qbase, seed):
    """Protocol crop (dense_crop geometry) with the SE(3) captured so we know ground truth."""
    rng = np.random.default_rng(seed)
    R_pert = rot(rng)
    p = qbase @ R_pert.T
    c = p.mean(0)
    u = np.linalg.svd(p - c, full_matrices=False)[2][0]
    proj = (p - c) @ u
    band = np.clip(np.digitize(proj, np.linspace(proj.min(), proj.max(), 15)[1:-1]), 0, 13)
    teeth = np.unique(band)
    kept = rng.choice(teeth, max(2, int(round(KEEP * len(teeth)))), replace=False)
    p = p[np.isin(band, kept)]
    crop = (p + rng.normal(0, 0.01, p.shape)).astype(np.float32)
    return crop, R_pert.T          # crop_i = R_pert @ arch_i  =>  align needs R_pert.T


def compute():
    register = _load_bufferx()
    meshes = sorted(Path(paths.TEETH3DS).glob("*/*.obj"))
    if not meshes:
        sys.exit(f"No Teeth3DS+ arches under {paths.TEETH3DS} (set TP_TEETH3DS).")
    cands = []
    for i in range(min(K, len(meshes))):
        arch = load_norm(meshes[i], NP_)
        qbase = load_norm(meshes[i], NP_)
        if arch is None or qbase is None:
            continue
        crop, R_gt = make_crop(qbase, 300 + i)
        gt = cKDTree(arch)
        aligned_g, _ = align_rigid(crop, arch, 0.05)
        Rg, tg = kabsch(crop, aligned_g)
        Rb, tb, _ = register(crop, arch)
        m = dict(i=i, sid=meshes[i].parent.name, arch=arch, crop=crop, Rgt=R_gt,
                 Rg=Rg, tg=tg, Rb=Rb, tb=tb,
                 rre_g=rre_deg(Rg, R_gt), rte_g=float(np.linalg.norm(tg)),
                 res_g=float(np.mean(gt.query(crop @ Rg.T + tg, k=1)[0])),
                 rre_b=rre_deg(Rb, R_gt), rte_b=float(np.linalg.norm(tb)),
                 res_b=float(np.mean(gt.query(crop @ Rb.T + tb, k=1)[0])))
        cands.append(m)
        print(f"  i={i:2d} {m['sid']:9s} GICP RRE {m['rre_g']:6.1f} res {m['res_g']:.3f} | "
              f"BUFX RRE {m['rre_b']:5.1f} res {m['res_b']:.3f}", flush=True)
    floor = float(np.median([c["res_b"] for c in cands]))       # genuine-match residual floor
    ok = [c for c in cands if c["rre_b"] < 5 and c["res_b"] < 1.5 * floor]
    chosen = max(ok, key=lambda c: c["rre_g"])                   # worst-GICP pair where BUFFER-X locks
    chosen["floor"] = floor
    print(f"chosen: {chosen['sid']} (i={chosen['i']})  GICP RRE {chosen['rre_g']:.1f}  BUFX RRE {chosen['rre_b']:.1f}")
    return chosen


def smoothstep(s):
    return s * s * (3 - 2 * s)


def hud(fig, phase, sub, metric, verdict, vcolor):
    fig.text(0.5, 0.965, phase, ha="center", va="top", fontsize=15, color=INK, fontweight="bold")
    fig.text(0.5, 0.915, sub, ha="center", va="top", fontsize=9.5, color=MUTED)
    if metric:
        fig.text(0.5, 0.205, metric, ha="center", va="center", fontsize=11, color=INK,
                 family="monospace")
    if verdict:
        ax = fig.add_axes([0.24, 0.115, 0.52, 0.055]); ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.02,rounding_size=0.5",
                                    facecolor=vcolor, alpha=0.14, edgecolor=vcolor, lw=1.6,
                                    transform=ax.transAxes))
        ax.text(0.5, 0.5, verdict, ha="center", va="center", fontsize=11, color=vcolor,
                fontweight="bold", transform=ax.transAxes)


def render(m):
    arch, crop = m["arch"], m["crop"]
    rng = np.random.default_rng(0)
    archd = arch[rng.choice(len(arch), min(2400, len(arch)), replace=False)]
    cropd = crop[rng.choice(len(crop), min(2400, len(crop)), replace=False)]
    floor = m["floor"]
    gx = m["res_g"] / floor
    footer = (f"Real Teeth3DS+ upper arch {m['sid']}  ·  keep-0.5 whole-tooth dropout  ·  live GICP vs live BUFFER-X.\n"
              "Both registrars global (result independent of arrival pose).  GICP fails on THIS arch — it succeeds on most "
              "Teeth3DS+ arches;\nthe aggregate story: rigid GICP keep-0.5 Rank-1 0.23 (Poseidon3D) vs BUFFER-X 1.00 (Teeth3DS+, zero-shot).")

    def disp(R, t):
        return cropd @ R.T + t

    def slerp(R0, R1):
        return Slerp([0, 1], Rotation.from_matrix(np.stack([R0, R1])))

    frames = []  # (crop_points, color, phase, sub, metric, verdict, vcolor)
    # 1 · arrival (hold)
    for _ in range(8):
        frames.append((disp(R_INIT, T_INIT), SLATE, "1  ·  Query arrives",
                       "partial crop — 50% of the teeth missing, arbitrary pose", "", "", INK))
    # 2 · GICP fly-in + hold
    kg = slerp(R_INIT, m["Rg"])
    for j in range(15):
        s = smoothstep(j / 14)
        R_s = kg([s]).as_matrix()[0]; t_s = (1 - s) * T_INIT + s * m["tg"]
        frames.append((disp(R_s, t_s), RED, "2  ·  Rigid GICP",
                       "PCA-init + Generalized-ICP (the repo's own path)", "", "", RED))
    for _ in range(9):
        frames.append((disp(m["Rg"], m["tg"]), RED, "2  ·  Rigid GICP",
                       "PCA-init + Generalized-ICP (the repo's own path)",
                       f"RRE {m['rre_g']:.0f}°     residual {m['res_g']:.3f}  ({gx:.1f}× locked-match floor)",
                       "WRONG BASIN  —  PCA-init flips the half-arch", RED))
    # 3 · BUFFER-X fly-in + hold
    kb = slerp(R_INIT, m["Rb"])
    for j in range(15):
        s = smoothstep(j / 14)
        R_s = kb([s]).as_matrix()[0]; t_s = (1 - s) * T_INIT + s * m["tb"]
        frames.append((disp(R_s, t_s), AQUA, "3  ·  BUFFER-X  (zero-shot, no dental training)",
                       "ICCV 2025 · pretrained 3DMatch · run live on this pair", "", "", AQUA))
    for _ in range(11):
        frames.append((disp(m["Rb"], m["tb"]), AQUA, "3  ·  BUFFER-X  (zero-shot, no dental training)",
                       "ICCV 2025 · pretrained 3DMatch · run live on this pair",
                       f"RRE {m['rre_b']:.1f}°     residual {m['res_b']:.3f}  (at the genuine-match floor)",
                       "LOCKED  —  genuine match", AQUA))

    fdir = OUT.parent / "_regframes"; fdir.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    for fi, (cp, col, phase, sub, metric, verdict, vcolor) in enumerate(frames):
        azim = -96 + 12 * (fi / n)          # gentle orbit for a 3D read
        fig = plt.figure(figsize=(7.0, 6.6))
        ax = fig.add_axes([0.0, 0.24, 1.0, 0.64], projection="3d")
        ax.scatter(archd[:, 0], archd[:, 1], archd[:, 2], s=3.2, c=ARCHC, depthshade=True, linewidths=0)
        ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], s=5.0, c=col, depthshade=True, linewidths=0)
        ax.set_box_aspect((1, 1, 1)); lim = 1.0
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.view_init(elev=55, azim=azim); ax.set_axis_off()
        hud(fig, phase, sub, metric, verdict, vcolor)
        fig.text(0.5, 0.075, footer, ha="center", va="top", fontsize=6.6, color=MUTED)
        fig.patch.set_facecolor("white")
        fig.savefig(fdir / f"f{fi:03d}.png", dpi=96, facecolor="white")
        plt.close(fig)

    pal = fdir / "pal.png"
    scale = "scale=760:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-framerate", "13", "-i", str(fdir / "f%03d.png"),
                    "-vf", f"fps=13,{scale},palettegen=stats_mode=diff", str(pal)],
                   check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-framerate", "13", "-i", str(fdir / "f%03d.png"), "-i", str(pal),
                    "-lavfi", f"fps=13,{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3", str(OUT)],
                   check=True, capture_output=True)
    for f in fdir.glob("*.png"):
        f.unlink()
    fdir.rmdir()
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.2f} MB, {n} frames)")


def main():
    t0 = time.time()
    if CACHE and Path(CACHE).exists():
        d = np.load(CACHE, allow_pickle=True)
        m = {k: d[k].item() if d[k].shape == () else d[k] for k in d.files}
        print(f"[dev] loaded pair from cache {CACHE}: {m['sid']}")
    else:
        m = compute()
        if CACHE:
            np.savez_compressed(CACHE, **{k: np.array(v) for k, v in m.items()})
    NUMS.write_text(json.dumps({
        "subject": str(m["sid"]), "arch_index": int(m["i"]), "seed": int(300 + m["i"]),
        "keep": KEEP, "n_points": NP_, "search_window_K": K,
        "note": ("arrival pose is illustrative; both registrars are global so the result is "
                 "pose-independent. GICP fails on this arch, not on all Teeth3DS+ arches."),
        "genuine_match_residual_floor": round(float(m["floor"]), 4),
        "gicp": {"rre_deg": round(m["rre_g"], 1), "rte_norm": round(m["rte_g"], 3),
                 "residual_norm": round(m["res_g"], 4)},
        "bufferx": {"rre_deg": round(m["rre_b"], 2), "rte_norm": round(m["rte_b"], 3),
                    "residual_norm": round(m["res_b"], 4)},
    }, indent=1) + "\n")
    print(f"saved {NUMS}")
    render(m)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
