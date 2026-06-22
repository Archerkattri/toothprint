"""Render a true periodontal change into image *pixels* (not just annotations).

The Gate-2 recall evaluation needs a follow-up image whose bone crest has
genuinely moved, so a pixel-reading detector (ViTPose) can actually measure the
change. Injecting the shift into the annotation alone is invisible to a real
detector: both timepoints would share identical pixels and the predicted change
would be structurally zero.

``render_crestal_change`` applies a smooth, *localized* displacement field that
translates the crest region along the bone vector by ``delta_px`` while leaving
the CEJ and surrounding anatomy fixed — a faithful image-space analogue of
localized crestal bone loss. The warp is a Gaussian-weighted inverse remap
(scipy ``map_coordinates``), restricted to a window around the crest for speed.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def render_crestal_change(
    image: np.ndarray,
    cej_mid: "list[float]",
    crest_mid: "list[float]",
    delta_px: float,
    *,
    width_px: "float | None" = None,
    sigma_along: "float | None" = None,
    order: int = 1,
) -> np.ndarray:
    """Return a copy of ``image`` with the bone margin receded by ``delta_px``.

    Models localized crestal bone loss: the whole bone-margin band around the
    crest is displaced **apically** along the CEJ->crest bone vector (positive =
    bone loss). The support is *anisotropic* — narrow along the bone vector
    (``sigma_along``) but **wide across the tooth** (``width_px``) — so the entire
    radiopaque bone-margin edge recedes, not just a small blob. This is what a
    learned, context-driven detector (ViTPose) actually responds to: an
    isotropic blob warp leaves most of the margin in place and is ignored.

    The shift direction matches ``inject_crestal_change``'s annotation edit, so
    image and label stay consistent.

    Args:
        image:       HxW or HxWxC uint8/float array (the baseline radiograph).
        cej_mid:     [x, y] CEJ midpoint in image pixels.
        crest_mid:   [x, y] crest midpoint in image pixels (the moving margin).
        delta_px:    apical shift in pixels. ``0`` returns an unchanged copy.
        width_px:    across-tooth extent (perp to bone vector) of the moved band;
                     defaults to ``2.5 * CEJ-crest distance`` so it spans the tooth.
        sigma_along: extent along the bone vector; defaults to
                     ``0.6 * CEJ-crest distance`` (clamped to >= 8 px).
        order:       spline interpolation order for resampling (1 = bilinear).

    Returns:
        Warped image, same shape and dtype as the input.
    """
    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValueError("image must be a 2D or 3D array")
    if delta_px == 0.0:
        return img.copy()

    cx, cy = float(cej_mid[0]), float(cej_mid[1])
    kx, ky = float(crest_mid[0]), float(crest_mid[1])
    dx, dy = kx - cx, ky - cy
    length = float(np.hypot(dx, dy))
    if length < 1e-9:
        raise ValueError("CEJ and crest midpoints coincide; bone vector undefined")
    ux, uy = dx / length, dy / length  # along bone (apical) unit vector
    nx, ny = -uy, ux  # across-tooth unit vector
    vx, vy = ux * float(delta_px), uy * float(delta_px)

    sa = sigma_along if sigma_along is not None else max(0.6 * length, 8.0)
    sp = (width_px if width_px is not None else 2.5 * length) / 2.0
    sp = max(sp, 12.0)

    h, w = img.shape[:2]
    # Window covers 3-sigma of the (anisotropic) support in both axes.
    r = int(3.0 * max(sa, sp)) + 1
    x0, x1 = max(0, int(kx) - r), min(w, int(kx) + r + 1)
    y0, y1 = max(0, int(ky) - r), min(h, int(ky) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return img.copy()

    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    rel_x, rel_y = xs - kx, ys - ky
    d_along = rel_x * ux + rel_y * uy
    d_perp = rel_x * nx + rel_y * ny
    weight = np.exp(-(d_along**2 / (2.0 * sa * sa) + d_perp**2 / (2.0 * sp * sp)))
    # Inverse map: output[q] samples input at q - v*w(q), moving the margin by v.
    src_x = xs - vx * weight
    src_y = ys - vy * weight

    out = img.copy()
    if img.ndim == 2:
        out[y0:y1, x0:x1] = map_coordinates(
            img, [src_y, src_x], order=order, mode="reflect"
        ).astype(img.dtype)
    else:
        for c in range(img.shape[2]):
            out[y0:y1, x0:x1, c] = map_coordinates(
                img[:, :, c], [src_y, src_x], order=order, mode="reflect"
            ).astype(img.dtype)
    return out
