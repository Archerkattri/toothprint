"""Surface reconstruction error metrics for dental map certification.

Measures how accurately a reconstructed point cloud matches a reference mesh,
using ICP alignment followed by Chamfer and point-to-surface distances.

All distances in millimetres.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class SurfaceError:
    """Surface reconstruction error between a reconstructed cloud and a reference."""
    chamfer_mm: float             # mean bidirectional Chamfer distance
    point_to_surface_mm: float    # mean point-to-surface distance (one-way: rec → ref)
    hausdorff_mm: float           # symmetric Hausdorff: max(directed rec->ref, ref->rec)
    n_reconstructed: int
    n_reference: int
    icp_iterations: int           # how many ICP iterations until convergence
    icp_converged: bool


def icp_align(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
    estimate_scale: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """Iterative Closest Point alignment of source to target.

    Parameters
    ----------
    source : (N, 3) array — points to transform
    target : (M, 3) array — fixed reference points
    max_iterations : int — stop after this many iterations
    tolerance : float — convergence threshold on mean distance change
    estimate_scale : bool — if True, fit a full **similarity** transform
        (rotation + translation + uniform scale, Umeyama 1991) instead of a
        rigid one. Required for scale-ambiguous reconstructions (e.g. DUSt3R /
        multiview neural backends recover geometry up to an unknown global
        scale), so the metric error reflects shape, not the arbitrary scale.

    Returns
    -------
    aligned_source : (N, 3) — source after ICP
    R : (3, 3) — rotation matrix (scaled by the fitted factor when estimate_scale)
    t : (3,) — translation vector
    n_iters : int — iterations performed
    converged : bool
    """
    src = source.copy().astype(float)
    tgt = target.astype(float)

    R_total = np.eye(3)
    t_total = np.zeros(3)
    prev_dist = np.inf
    n_iters = 0
    converged = False

    for i in range(max_iterations):
        n_iters = i + 1
        # Find closest target point for each source point
        diffs = src[:, np.newaxis, :] - tgt[np.newaxis, :, :]  # (N, M, 3)
        dists = np.linalg.norm(diffs, axis=2)  # (N, M)
        closest_idx = np.argmin(dists, axis=1)  # (N,)
        closest = tgt[closest_idx]  # (N, 3)

        mean_dist = float(np.mean(np.min(dists, axis=1)))

        # SVD-based alignment (Umeyama method)
        src_centroid = src.mean(axis=0)
        tgt_centroid = closest.mean(axis=0)
        src_c = src - src_centroid
        tgt_c = closest - tgt_centroid

        H = src_c.T @ tgt_c
        U, S, Vt = np.linalg.svd(H)
        # Ensure proper rotation (det = 1)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        if estimate_scale:
            # Umeyama uniform scale: s = trace(D @ S) / variance(src).
            var_src = float((src_c ** 2).sum() / src_c.shape[0])
            s = float(np.trace(D @ np.diag(S)) / src_c.shape[0] / var_src) if var_src > 0 else 1.0
            R = s * R
        t = tgt_centroid - R @ src_centroid

        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

        if abs(prev_dist - mean_dist) < tolerance:
            converged = True
            break
        prev_dist = mean_dist

    return src, R_total, t_total, n_iters, converged


def chamfer_distance(source: np.ndarray, target: np.ndarray) -> float:
    """Bidirectional mean Chamfer distance between two point clouds.

    Raises ``ValueError`` when either cloud is empty: the mean of an empty set
    is undefined, so fast-fail rather than masking it with a NaN sentinel.
    """
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("chamfer_distance requires non-empty source and target clouds")
    if source.shape[0] * target.shape[0] > 10_000_000:
        # Chunked source→target
        chunk = max(1, 10_000_000 // target.shape[0])
        min_dists_st = []
        for start in range(0, source.shape[0], chunk):
            block = source[start : start + chunk]
            d = np.sqrt(
                np.sum(
                    (block[:, np.newaxis, :] - target[np.newaxis, :, :]) ** 2, axis=2
                )
            )
            min_dists_st.append(np.min(d, axis=1))
        min_dists_st_arr = np.concatenate(min_dists_st)
        d_s_to_t = float(np.mean(min_dists_st_arr))

        # Chunked target→source
        chunk_t = max(1, 10_000_000 // source.shape[0])
        min_dists_ts = []
        for start in range(0, target.shape[0], chunk_t):
            block = target[start : start + chunk_t]
            d = np.sqrt(
                np.sum(
                    (block[:, np.newaxis, :] - source[np.newaxis, :, :]) ** 2, axis=2
                )
            )
            min_dists_ts.append(np.min(d, axis=1))
        min_dists_ts_arr = np.concatenate(min_dists_ts)
        d_t_to_s = float(np.mean(min_dists_ts_arr))
    else:
        # source → target
        diffs_st = source[:, np.newaxis, :] - target[np.newaxis, :, :]
        dists_st = np.sqrt(np.sum(diffs_st ** 2, axis=2))
        d_s_to_t = float(np.mean(np.min(dists_st, axis=1)))
        # target → source
        d_t_to_s = float(np.mean(np.min(dists_st, axis=0)))

    return (d_s_to_t + d_t_to_s) / 2.0


def _closest_point_on_segment(p: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Closest point to *p* on the line SEGMENT q0->q1 (clamped to the endpoints)."""
    seg = q1 - q0
    ll = float(np.dot(seg, seg))
    if ll < 1e-12:
        return q0
    u = float(np.clip(np.dot(p - q0, seg) / ll, 0.0, 1.0))
    return q0 + u * seg


def _directed_hausdorff(src: np.ndarray, dst: np.ndarray) -> float:
    """Directed Hausdorff distance: max over src of the nearest-dst distance."""
    if src.shape[0] * dst.shape[0] > 10_000_000:
        chunk = max(1, 10_000_000 // dst.shape[0])
        min_dists = []
        for start in range(0, src.shape[0], chunk):
            block = src[start : start + chunk]
            d = np.sqrt(np.sum((block[:, np.newaxis, :] - dst[np.newaxis, :, :]) ** 2, axis=2))
            min_dists.append(np.min(d, axis=1))
        return float(np.max(np.concatenate(min_dists)))
    diffs = src[:, np.newaxis, :] - dst[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    return float(np.max(np.min(dists, axis=1)))


def _point_to_triangle_distance(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> float:
    """Mean point-to-triangle distance using barycentric projection.

    For each point, finds the nearest triangle (by centroid), projects the
    point onto the triangle plane, clips to the triangle boundary if outside,
    and returns the distance to the clamped projection.

    Parameters
    ----------
    points : (N, 3)
    vertices : (M, 3)
    faces : (F, 3) int — indices into vertices
    """
    # Edge-case: no points to measure, or no triangles to measure against.
    # The mean over an empty set is undefined, so fast-fail rather than
    # masking it with NaN or indexing into an empty faces array.
    if points.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("_point_to_triangle_distance requires non-empty points and faces")

    # Triangle vertices
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Triangle centroids for coarse nearest-triangle lookup
    centroids = (v0 + v1 + v2) / 3.0  # (F, 3)

    # Edge vectors
    e1 = v1 - v0  # (F, 3)
    e2 = v2 - v0

    # Triangle normals (unnormalised)
    normals = np.cross(e1, e2)  # (F, 3)
    norm_lens = np.linalg.norm(normals, axis=1, keepdims=True)  # (F, 1)
    # Avoid division by zero for degenerate triangles
    safe_lens = np.where(norm_lens == 0, 1.0, norm_lens)
    unit_normals = normals / safe_lens  # (F, 3)

    distances = np.empty(len(points))

    for i, p in enumerate(points):
        # Find nearest triangle by centroid distance
        diff_c = centroids - p  # (F, 3)
        centroid_dists = np.linalg.norm(diff_c, axis=1)
        tri_idx = int(np.argmin(centroid_dists))

        a = v0[tri_idx]
        e1i = e1[tri_idx]
        e2i = e2[tri_idx]
        n = unit_normals[tri_idx]

        # Project p onto triangle plane
        w = p - a
        proj = p - np.dot(w, n) * n  # projected point on the plane

        # Compute barycentric coordinates of proj
        # Solve: proj = a + s*e1 + t*e2
        d11 = np.dot(e1i, e1i)
        d12 = np.dot(e1i, e2i)
        d22 = np.dot(e2i, e2i)
        wp = proj - a
        d1p = np.dot(e1i, wp)
        d2p = np.dot(e2i, wp)
        denom = d11 * d22 - d12 * d12

        if abs(denom) < 1e-12:
            # Degenerate triangle — fall back to nearest vertex
            dv = np.array([
                np.linalg.norm(p - v0[tri_idx]),
                np.linalg.norm(p - v1[tri_idx]),
                np.linalg.norm(p - v2[tri_idx]),
            ])
            distances[i] = float(np.min(dv))
            continue

        s = (d22 * d1p - d12 * d2p) / denom
        t = (d11 * d2p - d12 * d1p) / denom

        if s >= 0.0 and t >= 0.0 and s + t <= 1.0:
            # Interior / on-edge: the barycentric projection is the closest point.
            closest = a + s * e1i + t * e2i
        else:
            # Outside the triangle: the closest point lies on one of the three
            # edge segments. Independently clamping s,t and renormalising s+t>1
            # projects onto the infinite edge LINE, not the segment, giving a
            # wrong (too-small) distance — so project onto each segment instead.
            b = a + e1i  # second triangle vertex (v1)
            c = a + e2i  # third triangle vertex (v2)
            cands = (
                _closest_point_on_segment(p, a, b),
                _closest_point_on_segment(p, b, c),
                _closest_point_on_segment(p, c, a),
            )
            closest = min(cands, key=lambda q: float(np.dot(p - q, p - q)))
        distances[i] = float(np.linalg.norm(p - closest))

    return float(np.mean(distances))


def point_to_surface_distance(
    points: np.ndarray,
    reference_points: np.ndarray,
    reference_faces: np.ndarray | None = None,
) -> float:
    """Mean point-to-surface distance from points to reference.

    When reference_faces is None, uses nearest-neighbor distance (Chamfer
    half-distance) as a proxy. When faces are provided, uses proper
    barycentric projection onto the nearest triangle.

    Raises ``ValueError`` when there are no points to measure or no reference
    geometry: fast-fail rather than masking the undefined mean with NaN.
    """
    if points.shape[0] == 0 or reference_points.shape[0] == 0:
        raise ValueError("point_to_surface_distance requires non-empty points and reference")
    if reference_faces is None:
        # Nearest-neighbor proxy
        if points.shape[0] * reference_points.shape[0] > 10_000_000:
            chunk = max(1, 10_000_000 // reference_points.shape[0])
            min_dists = []
            for start in range(0, points.shape[0], chunk):
                block = points[start : start + chunk]
                d = np.sqrt(
                    np.sum(
                        (block[:, np.newaxis, :] - reference_points[np.newaxis, :, :]) ** 2,
                        axis=2,
                    )
                )
                min_dists.append(np.min(d, axis=1))
            return float(np.mean(np.concatenate(min_dists)))
        diffs = points[:, np.newaxis, :] - reference_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        return float(np.mean(np.min(dists, axis=1)))

    # Triangle projection (if faces provided)
    return _point_to_triangle_distance(points, reference_points, reference_faces)


def surface_error_mm(
    reconstructed: np.ndarray,
    reference: np.ndarray,
    reference_faces: np.ndarray | None = None,
    *,
    run_icp: bool = True,
    icp_max_iterations: int = 50,
    estimate_scale: bool = False,
) -> SurfaceError:
    """Compute full surface error between reconstructed and reference point clouds.

    Parameters
    ----------
    reconstructed : (N, 3) array — points from reconstruction pipeline
    reference : (M, 3) array — reference surface points (sampled from GT mesh)
    reference_faces : (F, 3) int array or None — triangle face indices for GT mesh
    run_icp : bool — align reconstructed to reference with ICP first
    icp_max_iterations : int — maximum ICP iterations
    estimate_scale : bool — fit a similarity transform (with uniform scale) in
        ICP. Use for scale-ambiguous neural reconstructions (DUSt3R/VGGT recover
        geometry up to an unknown global scale); the error then measures shape
        fidelity rather than the arbitrary reconstruction scale.
    """
    rec = reconstructed.astype(float)
    ref = reference.astype(float)

    # Fast-fail on empty input clouds: the metrics below are undefined for an
    # empty cloud, so raise rather than masking with a NaN sentinel.
    if rec.shape[0] == 0 or ref.shape[0] == 0:
        raise ValueError("surface_error_mm requires non-empty reconstructed and reference clouds")

    # Fast-fail on non-finite coordinates: NaN/inf rows would make ICP's
    # covariance SVD raise LinAlgError and poison every distance with NaN.
    # Surfacing the bad input is better than silently dropping the rows.
    if not np.isfinite(rec).all() or not np.isfinite(ref).all():
        raise ValueError("surface_error_mm requires all input coordinates to be finite")

    n_iters = 0
    converged = False
    if run_icp and len(rec) >= 3 and len(ref) >= 3:
        rec, _, _, n_iters, converged = icp_align(
            rec, ref, max_iterations=icp_max_iterations, estimate_scale=estimate_scale)

    cd = chamfer_distance(rec, ref)
    p2s = point_to_surface_distance(rec, ref, reference_faces)

    # Symmetric Hausdorff: max of both directed distances. A one-sided
    # (rec->ref) distance misses reference geometry that the reconstruction
    # failed to cover, understating the worst-case error.
    hausdorff = max(_directed_hausdorff(rec, ref), _directed_hausdorff(ref, rec))

    return SurfaceError(
        chamfer_mm=round(cd, 6),
        point_to_surface_mm=round(p2s, 6),
        hausdorff_mm=round(hausdorff, 6),
        n_reconstructed=len(reconstructed),
        n_reference=len(reference),
        icp_iterations=n_iters,
        icp_converged=converged,
    )
