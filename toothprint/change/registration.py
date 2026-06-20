"""Differential bone-level change measurement by sub-pixel image registration.

Regressing landmarks independently per timepoint compounds detector error into
~100 px of bone-level noise. Instead, measure the change *differentially*:
template-match the bone-margin patch between the two timepoints to sub-pixel
precision, referenced to a stationary crown patch so global acquisition
repositioning cancels, and project onto the apical bone vector. Stable pairs then
measure ~0.1 px, so even a sub-millimetre change is detectable.
"""
from __future__ import annotations

import numpy as np


def _patch(gray: np.ndarray, cx: float, cy: float, half: int):
    """Square patch (2*half) centred at (cx, cy) as float32, or None if out of bounds."""
    h, w = gray.shape
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return gray[y0:y1, x0:x1].astype(np.float32)


def _subpixel_peak(res: np.ndarray, mx: int, my: int) -> tuple[float, float]:
    """Parabolic sub-pixel refinement of an NCC peak at integer (mx, my)."""
    H, W = res.shape
    sx, sy = float(mx), float(my)
    if 1 <= mx < W - 1:
        l, m, r = res[my, mx - 1], res[my, mx], res[my, mx + 1]
        dd = l - 2 * m + r
        if abs(dd) > 1e-9:
            sx += 0.5 * (l - r) / dd
    if 1 <= my < H - 1:
        l, m, r = res[my - 1, mx], res[my, mx], res[my + 1, mx]
        dd = l - 2 * m + r
        if abs(dd) > 1e-9:
            sy += 0.5 * (l - r) / dd
    return sx, sy


def measure_displacement(g0: np.ndarray, g1: np.ndarray, center, half: int = 20,
                         search: int = 70):
    """Sub-pixel (dx, dy) displacement of content at ``center`` from g0 to g1.

    Returns ``((dx, dy), response)`` or ``None`` if a window is out of bounds.
    """
    import cv2

    tmpl = _patch(g0, center[0], center[1], half)
    region = _patch(g1, center[0], center[1], half + search)
    if tmpl is None or region is None:
        return None
    res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    sx, sy = _subpixel_peak(res, maxloc[0], maxloc[1])
    return (sx - search, sy - search), float(maxv)


def measure_change(g0: np.ndarray, g1: np.ndarray, reference_center, crest_center,
                   bone_unit, half: int = 20, search: int = 70):
    """Reference-cancelled apical bone-level change (px) between two timepoints.

    The crest displacement is measured relative to a stationary reference (crown)
    patch, so global motion cancels; positive = bone loss. Returns
    ``(change, response)`` or ``None`` if a patch is out of bounds.
    """
    crest = measure_displacement(g0, g1, crest_center, half, search)
    ref = measure_displacement(g0, g1, reference_center, half, search)
    if crest is None or ref is None:
        return None
    (cdx, cdy), cresp = crest
    (rdx, rdy), rresp = ref
    rel = (cdx - rdx) * bone_unit[0] + (cdy - rdy) * bone_unit[1]
    return float(rel), float(min(cresp, rresp))


def measure_change_search(g0, g1, reference_center, crest_center, bone_unit, offsets,
                          half: int = 20, search: int = 70):
    """Largest apical change over candidate crest positions along the bone vector.

    A coarse localisation rarely lands the patch exactly on the bone margin;
    sliding it over ``offsets`` (px) and taking the most-apical displacement finds
    the moving margin. Returns ``(change, response)`` of the best candidate, or
    ``None``.
    """
    best = None
    for t in offsets:
        c = (crest_center[0] + t * bone_unit[0], crest_center[1] + t * bone_unit[1])
        out = measure_change(g0, g1, reference_center, c, bone_unit, half, search)
        if out is None:
            continue
        if best is None or out[0] > best[0]:
            best = out
    return best
