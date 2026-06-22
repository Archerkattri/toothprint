"""2D dental identification from radiograph landmark constellations.

The spatial pattern of a person's per-tooth landmarks (CEJ, bone crest, apex) is
individual. Two constellations are scale-normalised (so projection magnification
cancels) and aligned with a rigid ICP; the gallery subject with the smallest
residual is the identity. Free-scale alignment is avoided on purpose — it lets a
query collapse onto a gallery cluster and gives impostors a spurious zero residual.
"""

from __future__ import annotations

import numpy as np

from toothprint.geometry import umeyama


def constellation(annotation: dict) -> np.ndarray:
    """Flatten a radiograph annotation into an (M, 2) landmark point set."""
    pts = []
    for tooth in annotation.get("teeth", []):
        for field in ("cej", "crest_line", "apex"):
            for p in tooth.get(field) or []:
                pts.append([float(p[0]), float(p[1])])
    if not pts:
        raise ValueError("annotation has no landmarks to form a constellation")
    return np.asarray(pts, dtype=np.float64)


def _normalize(c: np.ndarray):
    """Zero-mean, unit-RMS-radius normalisation; returns (normed, rms_radius)."""
    c = np.asarray(c, dtype=np.float64)
    centered = c - c.mean(0)
    rms = float(np.sqrt((centered**2).sum(axis=1).mean()))
    rms = rms if rms > 1e-9 else 1.0
    return centered / rms, rms


def icp_residual(
    query: np.ndarray, gallery: np.ndarray, *, iters: int = 30, tol: float = 1e-6
) -> float:
    """Scale-normalised rigid-ICP residual RMS aligning ``query`` onto ``gallery`` (px)."""
    src, _ = _normalize(query)
    dst, dst_rms = _normalize(gallery)
    prev = np.inf
    rms = float("inf")
    for _ in range(iters):
        d2 = ((src[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)
        nn = d2.argmin(axis=1)
        rms = float(np.sqrt(d2[np.arange(len(src)), nn].mean()))
        A, t = umeyama(src, dst[nn], with_scale=False)
        src = src @ A.T + t
        if abs(prev - rms) < tol:
            break
        prev = rms
    return rms * dst_rms


def identify(query_constellation: np.ndarray, gallery_constellations) -> np.ndarray:
    """Residual of one query constellation against every gallery constellation."""
    return np.array(
        [icp_residual(query_constellation, g) for g in gallery_constellations]
    )
