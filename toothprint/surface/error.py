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


def noise_floor_sq(stable_pairs) -> float:
    """Reconstruction-noise power ``E[mean ||t1 - t0||^2]`` from stable pairs.

    ``stable_pairs`` is an iterable of ``(cloud_t0, cloud_t1)`` re-scan pairs with
    *no* real change between them (in point correspondence). The return value is
    the mean per-point squared displacement — the noise power that
    :func:`surface_displacement` subtracts to de-bias a change measurement.
    """
    vals = []
    for a, b in stable_pairs:
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        if a.shape != b.shape or a.shape[0] == 0:
            raise ValueError("each stable pair must be non-empty corresponded clouds")
        vals.append(float(((b - a) ** 2).sum(axis=1).mean()))
    if not vals:
        raise ValueError("at least one stable pair is required")
    return float(np.mean(vals))


def surface_displacement(cloud_t0: np.ndarray, cloud_t1: np.ndarray, *,
                         noise_floor_sq: float = 0.0) -> float:
    """De-biased mean surface displacement (mm) between two corresponded timepoints.

    The per-point displacement is ``v_i = t1_i - t0_i`` (clouds in correspondence,
    e.g. established by registration upstream). Aggregating as ``mean_i ||v_i||``
    *rectifies* zero-mean reconstruction noise into a positive bias ~k*sigma that
    inflates the certificate's conformal radius and hides small changes. This
    instead estimates the displacement magnitude in noise-power-corrected (squared)
    space::

        sqrt(max(0, mean_i ||v_i||^2 - noise_floor_sq))

    because ``E[mean ||v||^2] = (true displacement power) + (noise power)``. With
    ``noise_floor_sq`` = :func:`noise_floor_sq` of the *stable* calibration pairs,
    a no-change pair returns ~0 and a real displacement of magnitude m returns ~m,
    largely independent of the noise level — so the conformal certificate
    (calibrated on the de-biased stable residuals) keeps its sensitivity as
    reconstruction noise grows.

    Caveat: the residual spread of this estimate shrinks with the point count only
    when the reconstruction noise is spatially *incoherent*; spatially correlated
    error (the realistic case) does not average out, so the de-biasing gain is an
    upper bound — see the correlated-noise ablation in the evaluation.
    """
    a = np.asarray(cloud_t0, float)
    b = np.asarray(cloud_t1, float)
    if a.shape != b.shape:
        raise ValueError("surface_displacement requires corresponded clouds of equal shape")
    if a.shape[0] == 0:
        raise ValueError("surface_displacement requires non-empty clouds")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("surface_displacement requires finite coordinates")
    s2 = float(((b - a) ** 2).sum(axis=1).mean())
    return float(np.sqrt(max(0.0, s2 - float(noise_floor_sq))))


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
