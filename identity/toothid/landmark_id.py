"""2D dental biometric identification from radiograph landmark constellations.

The spatial pattern of a person's per-tooth landmarks (CEJ, bone crest, apex) is
individual. This recognises a person by aligning their landmark *constellation*
to each gallery constellation with a 2D similarity ICP (rotation + translation +
scale, so projection magnification cancels) and taking the residual RMS: the
gallery person with the smallest residual is the identity — the 2D analogue of
the 3D point-cloud method (smallest registration RMSE wins).
"""
from __future__ import annotations

import numpy as np


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


def _umeyama_2d(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity (sR, t) mapping src onto dst (paired points)."""
    sc = src.mean(0)
    dc = dst.mean(0)
    s = src - sc
    d = dst - dc
    H = s.T @ d / len(src)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, np.linalg.det(Vt.T @ U.T)])
    R = Vt.T @ D @ U.T
    scale = 1.0
    if with_scale:
        var = (s ** 2).sum() / len(src)
        scale = float(np.trace(D @ np.diag(S)) / var) if var > 0 else 1.0
    t = dc - scale * R @ sc
    return scale * R, t


def _normalize(c: np.ndarray):
    """Zero-mean, unit-RMS-radius normalization; returns (normed, rms_radius)."""
    c = np.asarray(c, dtype=np.float64)
    centered = c - c.mean(0)
    rms = float(np.sqrt((centered ** 2).sum(axis=1).mean()))
    rms = rms if rms > 1e-9 else 1.0
    return centered / rms, rms


def icp_residual_2d(query: np.ndarray, gallery: np.ndarray, *, iters: int = 30,
                    tol: float = 1e-6) -> float:
    """Rigid-ICP residual RMS aligning ``query`` onto ``gallery`` (px).

    Both constellations are first normalized to unit scale (so projection
    magnification cancels) and aligned with a **rigid** (rotation+translation,
    no free scale) ICP — free scale would let the query collapse onto a gallery
    cluster and give impostors a spurious zero residual. The normalized residual
    is reported back in pixels via the gallery's original scale.
    """
    src, _ = _normalize(query)
    dst, dst_rms = _normalize(gallery)
    prev = np.inf
    rms = float("inf")
    for _ in range(iters):
        d2 = ((src[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)
        nn = d2.argmin(axis=1)
        rms = float(np.sqrt(d2[np.arange(len(src)), nn].mean()))
        A, t = _umeyama_2d(src, dst[nn], with_scale=False)
        src = src @ A.T + t
        if abs(prev - rms) < tol:
            break
        prev = rms
    return rms * dst_rms


def residual_matrix(query_constellations, gallery_constellations) -> np.ndarray:
    """RMS residual of every query constellation against every gallery one."""
    q = list(query_constellations)
    g = list(gallery_constellations)
    out = np.zeros((len(q), len(g)))
    for i, qc in enumerate(q):
        for j, gc in enumerate(g):
            out[i, j] = icp_residual_2d(qc, gc)
    return out
