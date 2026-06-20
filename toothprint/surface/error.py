"""Surface error between two point clouds, with scale-aware alignment.

Multiview neural reconstructions recover geometry up to an unknown global scale,
so alignment optionally fits a similarity transform (Umeyama, with scale) before
measuring error. All metrics are pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceError:
    chamfer_mm: float       # mean bidirectional nearest-point distance
    hausdorff_mm: float     # symmetric worst-case distance
    rms_mm: float           # RMS of source->target nearest-point distances
    n_source: int
    n_target: int
    icp_iterations: int


def _nearest(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-source nearest-neighbour distances to dst (brute force)."""
    d2 = ((src[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean bidirectional nearest-point distance."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise ValueError("chamfer_distance requires non-empty clouds")
    return float((_nearest(a, b).mean() + _nearest(b, a).mean()) / 2.0)


def icp_align(source: np.ndarray, target: np.ndarray, *, max_iterations: int = 50,
              tolerance: float = 1e-5, estimate_scale: bool = False):
    """Align source onto target with ICP; returns ``(aligned, n_iters)``.

    ``estimate_scale=True`` fits a full similarity transform (Umeyama) each
    iteration — required for scale-ambiguous reconstructions.
    """
    src = np.asarray(source, float).copy()
    tgt = np.asarray(target, float)
    prev = np.inf
    n_iters = 0
    for i in range(max_iterations):
        n_iters = i + 1
        d2 = ((src[:, None, :] - tgt[None, :, :]) ** 2).sum(axis=2)
        nn = d2.argmin(axis=1)
        closest = tgt[nn]
        mean_d = float(np.sqrt(d2[np.arange(len(src)), nn]).mean())
        sc, dc = src.mean(0), closest.mean(0)
        s, d = src - sc, closest - dc
        H = s.T @ d
        U, S, Vt = np.linalg.svd(H)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        if estimate_scale:
            var = float((s ** 2).sum() / len(src))
            scale = float(np.trace(D @ np.diag(S)) / len(src) / var) if var > 0 else 1.0
            R = scale * R
        t = dc - R @ sc
        src = src @ R.T + t
        if abs(prev - mean_d) < tolerance:
            break
        prev = mean_d
    return src, n_iters


def surface_error(reconstructed: np.ndarray, reference: np.ndarray, *,
                  run_icp: bool = True, estimate_scale: bool = False) -> SurfaceError:
    """Full surface error between a reconstructed and a reference point cloud."""
    rec = np.asarray(reconstructed, float)
    ref = np.asarray(reference, float)
    if rec.shape[0] == 0 or ref.shape[0] == 0:
        raise ValueError("surface_error requires non-empty clouds")
    if not (np.isfinite(rec).all() and np.isfinite(ref).all()):
        raise ValueError("surface_error requires finite coordinates")
    n_iters = 0
    if run_icp and len(rec) >= 3 and len(ref) >= 3:
        rec, n_iters = icp_align(rec, ref, estimate_scale=estimate_scale)
    fwd = _nearest(rec, ref)
    bwd = _nearest(ref, rec)
    return SurfaceError(
        chamfer_mm=round(float((fwd.mean() + bwd.mean()) / 2.0), 6),
        hausdorff_mm=round(float(max(fwd.max(), bwd.max())), 6),
        rms_mm=round(float(np.sqrt((fwd ** 2).mean())), 6),
        n_source=len(reconstructed),
        n_target=len(reference),
        icp_iterations=n_iters,
    )
