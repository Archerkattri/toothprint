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
        left, mid, right = res[my, mx - 1], res[my, mx], res[my, mx + 1]
        dd = left - 2 * mid + right
        if abs(dd) > 1e-9:
            sx += 0.5 * (left - right) / dd
    if 1 <= my < H - 1:
        left, mid, right = res[my - 1, mx], res[my, mx], res[my + 1, mx]
        dd = left - 2 * mid + right
        if abs(dd) > 1e-9:
            sy += 0.5 * (left - right) / dd
    return sx, sy


def measure_displacement(
    g0: np.ndarray, g1: np.ndarray, center, half: int = 20, search: int = 70
):
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


def measure_change(
    g0: np.ndarray,
    g1: np.ndarray,
    reference_center,
    crest_center,
    bone_unit,
    half: int = 20,
    search: int = 70,
):
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


def _bilinear(gray: np.ndarray, x: float, y: float):
    """Bilinearly-sampled intensity at sub-pixel ``(x, y)``, or None if out of bounds."""
    h, w = gray.shape
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    if x0 < 0 or y0 < 0 or x0 + 1 >= w or y0 + 1 >= h:
        return None
    fx, fy = x - x0, y - y0
    return float(
        gray[y0, x0] * (1 - fx) * (1 - fy)
        + gray[y0, x0 + 1] * fx * (1 - fy)
        + gray[y0 + 1, x0] * (1 - fx) * fy
        + gray[y0 + 1, x0 + 1] * fx * fy
    )


def snap_to_margin(
    gray: np.ndarray, center, bone_unit, span: float = 40.0, step: float = 2.0
):
    """Refine ``center`` along ±``bone_unit`` onto the strongest apical intensity
    edge — the bone margin itself.

    A learned detector localises the crest only coarsely (~35 px on DenPAR), so its
    patch sits off the radiopaque margin and *attenuates* the measured change. The
    margin is the dominant intensity step along the bone vector; snapping the centre
    to the peak of ``|dI/du|`` lands the patch on it — consistently for stable and
    changed pairs alike, which both tightens the noise floor and recovers signal.
    Returns the snapped ``(x, y)`` (the original centre if no edge is in range).
    """
    ux, uy = float(bone_unit[0]), float(bone_unit[1])

    def grad_at(cx, cy):
        a = _bilinear(gray, cx - step * ux, cy - step * uy)
        b = _bilinear(gray, cx + step * ux, cy + step * uy)
        return None if (a is None or b is None) else abs(b - a)

    best = (float(center[0]), float(center[1]))
    g_center = grad_at(*best)
    best_g = g_center if g_center is not None else -1.0  # anchor on the centre itself
    t = -span
    while t <= span:
        cx, cy = center[0] + t * ux, center[1] + t * uy
        g = grad_at(cx, cy)
        if g is not None and g > best_g:
            best_g, best = g, (cx, cy)
        t += 1.0
    return best


def fit_global_motion(
    g0: np.ndarray,
    g1: np.ndarray,
    anchors,
    half: int = 20,
    search: int = 70,
    min_response: float = 0.3,
):
    """Affine global-motion model ``t0 → t1`` from stationary anchor patches.

    A single reference patch only cancels a global *translation*; real
    repositioning between visits adds rotation, magnification, and projection-angle
    (perspective) change, under which different image locations move by different
    amounts. Template-matching several stationary crown ``anchors`` and fitting a
    least-squares affine recovers that full motion field, so it can be evaluated
    *at the crest* (see :func:`measure_change_anchored`) rather than assumed equal
    to one far reference.

    Returns a ``(3, 2)`` matrix ``P`` mapping ``[x, y, 1] → [x', y']`` (the t1
    location of t0 content), or ``None`` if fewer than 3 anchors match reliably
    (an affine needs 3 non-collinear correspondences).
    """
    src, dst = [], []
    for c in anchors:
        out = measure_displacement(g0, g1, c, half, search)
        if out is None:
            continue
        (dx, dy), resp = out
        if resp < min_response:
            continue
        src.append([c[0], c[1]])
        dst.append([c[0] + dx, c[1] + dy])
    if len(src) < 3:
        return None
    A = np.column_stack([np.asarray(src, float), np.ones(len(src))])
    P, *_ = np.linalg.lstsq(A, np.asarray(dst, float), rcond=None)
    return P


def measure_change_anchored(
    g0,
    g1,
    anchors,
    crest_center,
    bone_unit,
    half: int = 20,
    search: int = 70,
    min_response: float = 0.3,
):
    """Local apical bone change with a multi-anchor *affine* global-motion model.

    Cancels the global motion **evaluated at the crest location** — so arbitrary
    repositioning (large translation, rotation, magnification, modest
    projection-angle change) is removed, not just a pure translation. Returns
    ``(change, response)`` or ``None`` if the motion fit or the crest match fails.
    """
    P = fit_global_motion(g0, g1, anchors, half, search, min_response)
    crest = measure_displacement(g0, g1, crest_center, half, search)
    if P is None or crest is None:
        return None
    (cdx, cdy), cresp = crest
    pred = (
        np.asarray([crest_center[0], crest_center[1], 1.0]) @ P
    )  # global t1 location at crest
    gdx, gdy = pred[0] - crest_center[0], pred[1] - crest_center[1]
    rel = (cdx - gdx) * bone_unit[0] + (cdy - gdy) * bone_unit[1]
    return float(rel), float(cresp)


def measure_change_search(
    g0,
    g1,
    reference_center,
    crest_center,
    bone_unit,
    offsets,
    half: int = 20,
    search: int = 70,
    min_response: float = 0.0,
):
    """Largest apical change over *reliable* candidate crest positions.

    A coarse localisation rarely lands the patch exactly on the bone margin;
    sliding it over ``offsets`` (px) and taking the most-apical displacement finds
    the moving margin. ``min_response`` gates candidates by NCC match reliability,
    so a spurious large displacement from a poorly-matched (textureless) patch
    can't set the value and inflate the noise floor — only well-matched candidates
    compete for the maximum, which lets the certificate detect smaller changes. If
    no candidate clears the gate the most-reliable one is returned. Returns
    ``(change, response)`` of the best candidate, or ``None``.
    """
    best = None  # max change among reliable candidates
    fallback = None  # most-reliable candidate, if none clear the gate
    for t in offsets:
        c = (crest_center[0] + t * bone_unit[0], crest_center[1] + t * bone_unit[1])
        out = measure_change(g0, g1, reference_center, c, bone_unit, half, search)
        if out is None:
            continue
        if fallback is None or out[1] > fallback[1]:
            fallback = out
        if out[1] >= min_response and (best is None or out[0] > best[0]):
            best = out
    return best if best is not None else fallback
