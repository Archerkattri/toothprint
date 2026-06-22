"""Shared 2D/3D geometry primitives."""

from __future__ import annotations

import numpy as np


def mean_point(points) -> np.ndarray:
    """Centroid of a sequence of points."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("cannot average an empty point set")
    return pts.mean(axis=0)


def distance(a, b) -> float:
    """Euclidean distance between two points."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def umeyama(src: np.ndarray, dst: np.ndarray, *, with_scale: bool = True):
    """Least-squares similarity transform (sR, t) mapping ``src`` onto ``dst``.

    Works in any dimension; ``with_scale=False`` returns a rigid transform.
    Returns ``(A, t)`` where the mapped points are ``src @ A.T + t``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    dim = src.shape[1]
    sc = src.mean(0)
    dc = dst.mean(0)
    s = src - sc
    d = dst - dc
    H = s.T @ d / len(src)
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(dim)
    D[-1, -1] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    scale = 1.0
    if with_scale:
        var = (s**2).sum() / len(src)
        scale = float(np.trace(D @ np.diag(S)) / var) if var > 0 else 1.0
    t = dc - scale * R @ sc
    return scale * R, t
