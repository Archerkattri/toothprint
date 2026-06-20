#!/usr/bin/env python3
"""Probe: measure bone-margin change by sub-pixel image registration (phase corr).

Instead of regressing landmarks independently per timepoint (which compounds the
detector's ~35px error), register the bone-margin patch between t0 and t1 and
read the displacement directly. Tests whether this recovers the rendered crestal
shift with low variance — the basis for a precise end-to-end change measurement.
"""
from __future__ import annotations

import argparse
import cv2
import numpy as np
from PIL import Image

from dcc.data.denpar_adapter import RealDenparAdapter
from dcc.geometry import mean_point
from dcc.perturb.image_change import render_crestal_change
from dcc.benchmark.real_image_eval import acquire


def _refine_to_margin(gray, center, bone_unit, search=35):
    """Snap a coarse crest estimate to the strongest intensity edge along the
    apical bone vector (the radiopaque bone margin)."""
    from scipy.ndimage import map_coordinates
    ts = np.arange(-search, search + 1, dtype=np.float64)
    xs = center[0] + ts * bone_unit[0]
    ys = center[1] + ts * bone_unit[1]
    prof = map_coordinates(gray.astype(np.float64), [ys, xs], order=1, mode="reflect")
    # smooth then take the largest-magnitude gradient along the profile
    k = np.array([1, 4, 6, 4, 1], dtype=np.float64); k /= k.sum()
    sm = np.convolve(prof, k, mode="same")
    grad = np.abs(np.gradient(sm))
    t_star = ts[int(np.argmax(grad))]
    return np.array([center[0] + t_star * bone_unit[0], center[1] + t_star * bone_unit[1]])


def _patch(gray, cx, cy, half):
    h, w = gray.shape
    x0, y0 = int(cx) - half, int(cy) - half
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return gray[y0:y1, x0:x1].astype(np.float32)


def measure_change(g0, g1, center, bone_unit, half=20, search=70):
    """Apical bone-margin displacement (px) from t0->t1 via template matching.

    A small template centred on the bone margin in t0 is searched in t1 over a
    +/-search window; the NCC peak (sub-pixel via parabolic interpolation) gives
    the displacement, projected onto the apical bone vector (the well-constrained
    direction; the along-margin component is aperture-ambiguous and discarded).
    """
    tmpl = _patch(g0, center[0], center[1], half)
    region = _patch(g1, center[0], center[1], half + search)
    if tmpl is None or region is None:
        return None
    res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    mx, my = maxloc
    # Sub-pixel refine via parabola on the NCC peak in each axis.
    def _sub(c, axis):
        H, W = res.shape
        if axis == 0 and 1 <= c < W - 1:
            l, m, r = res[my, c - 1], res[my, c], res[my, c + 1]
        elif axis == 1 and 1 <= c < H - 1:
            l, m, r = res[c - 1, mx], res[c, mx], res[c + 1, mx]
        else:
            return float(c)
        denom = (l - 2 * m + r)
        return float(c) + (0.5 * (l - r) / denom if abs(denom) > 1e-9 else 0.0)
    sx, sy = _sub(mx, 0), _sub(my, 1)
    # Displacement of t1 content relative to t0 template: match offset minus the
    # zero-shift centre (which is `search`).
    return (sx - search, sy - search), float(maxv)


def measure_bonelevel_change(g0, g1, cej_c, crest_c, bone_unit, half=20, search=70):
    """CEJ-referenced apical bone-margin change (px), robust to global motion.

    Measures the crest-margin displacement and the CEJ displacement separately;
    a global acquisition shift moves both equally and cancels, leaving only the
    real crest recession projected onto the bone vector.
    """
    crest = measure_change(g0, g1, crest_c, bone_unit, half, search)
    cej = measure_change(g0, g1, cej_c, bone_unit, half, search)
    if crest is None or cej is None:
        return None
    (cdx, cdy), cresp = crest
    (jdx, jdy), jresp = cej
    rel = ((cdx - jdx) * bone_unit[0] + (cdy - jdy) * bone_unit[1])
    return float(rel), float(min(cresp, jresp))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/denpar/extracted/Dataset")
    p.add_argument("--n-teeth", type=int, default=120)
    p.add_argument("--deltas", default="0,15,30,60")
    p.add_argument("--acq-noise", type=float, default=3.0)
    p.add_argument("--half", type=int, default=20)
    p.add_argument("--detector", action="store_true",
                   help="Use ViTPose to localize the patches instead of GT")
    p.add_argument("--weights", default="outputs/vitpose_detector/checkpoint_best.pt")
    args = p.parse_args()

    det = None
    if args.detector:
        from dcc.landmarks.vitpose_detector import ViTPoseLandmarkDetector, predict_tooth
        det = ViTPoseLandmarkDetector(args.weights, device="cuda")

    deltas = [float(x) for x in args.deltas.split(",")]
    recs = list(RealDenparAdapter(args.data).records("test"))
    rng = np.random.default_rng(0)

    def _global_shift(im, sx, sy):
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    measured = {d: [] for d in deltas}
    n = 0
    for rec in recs:
        if n >= args.n_teeth:
            break
        img = np.array(Image.open(rec.image_path).convert("RGB"))
        for tooth in rec.annotation_dict.get("teeth", []):
            if n >= args.n_teeth:
                break
            if not (tooth.get("cej") and tooth.get("crest_line")):
                continue
            # GT anatomy: where the real change happens + true apical direction.
            gt_cej = np.array(mean_point(tooth["cej"]))
            gt_crest = np.array(mean_point(tooth["crest_line"]))
            v = gt_crest - gt_cej
            L = np.linalg.norm(v)
            if L < 1e-6:
                continue
            u = v / L
            # Patch centres: the detector's (imperfect) localization, or GT.
            if det is not None:
                from dcc.landmarks.vitpose_detector import predict_tooth
                pred = predict_tooth(det, img, tooth, n_tta=9, rng=rng)
                if pred is None or not pred.get("cej") or not pred.get("crest_line"):
                    continue
                cej_c = np.array(mean_point(pred["cej"]))
                crest_c = np.array(mean_point(pred["crest_line"]))
                # refine the coarse crest to the actual bone-margin edge in t0
                crest_c = _refine_to_margin(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), crest_c, u)
            else:
                cej_c, crest_c = gt_cej, gt_crest
            ok, local = True, {}
            for d in deltas:
                # acquisition repositioning: a global shift applied at t1
                gx, gy = rng.uniform(-6, 6), rng.uniform(-6, 6)
                warp = render_crestal_change(img, gt_cej.tolist(), gt_crest.tolist(), d)
                t1 = _global_shift(acquire(warp, rng, args.acq_noise), gx, gy)
                g1 = cv2.cvtColor(t1, cv2.COLOR_RGB2GRAY)
                g0 = cv2.cvtColor(acquire(img, rng, args.acq_noise), cv2.COLOR_RGB2GRAY)
                r = measure_bonelevel_change(g0, g1, cej_c, crest_c, u, half=args.half)
                if r is None:
                    ok = False
                    break
                local[d] = r[0]
            if ok:
                for d in deltas:
                    measured[d].append(local[d])
                n += 1

    print(f"\nCEJ-referenced change measurement over {n} teeth, GT localization + global shift\n")
    print(f"  {'delta_px':>9} {'median':>9} {'IQR':>9} {'mean':>9}")
    for d in deltas:
        a = np.array(measured[d])
        iqr = np.percentile(a, 75) - np.percentile(a, 25)
        print(f"  {d:>9.0f} {np.median(a):>9.2f} {iqr:>9.2f} {a.mean():>9.2f}")


if __name__ == "__main__":
    main()
