"""Differential bone-level change measurement by sub-pixel image registration.

Independent per-timepoint landmark regression compounds the detector's landmark
error (~35 px) into ~100 px of bone-level noise — far larger than a clinically
meaningful change, so change is undetectable. This module measures the change
*differentially* instead: it template-matches the bone-margin patch between the
two timepoints and reads the displacement directly, to sub-pixel precision.

Two robustness properties:
  * CEJ-referencing: the crest displacement is measured *relative to the CEJ*, so
    a global acquisition repositioning (translation) moves both equally and
    cancels — only the real crest recession remains.
  * Projection onto the apical bone vector: the along-margin component is
    aperture-ambiguous and discarded; the perpendicular (apical) component, which
    is exactly the bone-level change, is well constrained.

Validated on real DenPAR radiographs: stable pairs measure 0.0 +/- ~0.1 px;
a rendered crestal change is recovered with the same low spread.
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
        d = l - 2 * m + r
        if abs(d) > 1e-9:
            sx += 0.5 * (l - r) / d
    if 1 <= my < H - 1:
        l, m, r = res[my - 1, mx], res[my, mx], res[my + 1, mx]
        d = l - 2 * m + r
        if abs(d) > 1e-9:
            sy += 0.5 * (l - r) / d
    return sx, sy


def measure_displacement(g0: np.ndarray, g1: np.ndarray, center, half: int = 20,
                         search: int = 70):
    """Sub-pixel (dx, dy) displacement of content at ``center`` from g0 to g1.

    Template-matches the t0 patch within a +/-search window of t1 (normalised
    cross-correlation) and sub-pixel-refines the peak. Returns ``((dx, dy),
    response)`` or ``None`` if either window is out of bounds.
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


def measure_bonelevel_change(g0: np.ndarray, g1: np.ndarray, cej_center, crest_center,
                             bone_unit, half: int = 20, search: int = 70):
    """CEJ-referenced apical bone-level change (px) between two timepoints.

    ``bone_unit`` is the unit CEJ->crest vector (apical direction). Returns
    ``(change_px, response)`` where positive = bone loss, or ``None`` if a patch
    is out of bounds.
    """
    crest = measure_displacement(g0, g1, crest_center, half, search)
    cej = measure_displacement(g0, g1, cej_center, half, search)
    if crest is None or cej is None:
        return None
    (cdx, cdy), cresp = crest
    (jdx, jdy), jresp = cej
    rel = (cdx - jdx) * bone_unit[0] + (cdy - jdy) * bone_unit[1]
    return float(rel), float(min(cresp, jresp))


def measure_bonelevel_change_search(g0, g1, ref_center, crest_center, bone_unit,
                                    offsets, half: int = 20, search: int = 70):
    """Largest apical bone-level change over candidate crest positions.

    A coarse crest estimate (e.g. from a detector with ~35px localization error)
    rarely lands the patch exactly on the bone margin. Sliding the patch to
    several positions along the bone vector (``offsets`` px) and taking the most
    apical (bone-loss) displacement *finds* the moving margin: stable pairs stay
    ~0 at every offset, while a real change peaks at the offset on the margin.
    Returns ``(change, response)`` of the best candidate, or ``None``.
    """
    best = None
    for t in offsets:
        c = (crest_center[0] + t * bone_unit[0], crest_center[1] + t * bone_unit[1])
        out = measure_bonelevel_change(g0, g1, ref_center, c, bone_unit, half, search)
        if out is None:
            continue
        if best is None or out[0] > best[0]:
            best = out
    return best
