"""3D dental identification from intraoral-scan point clouds.

A person's dental arch (crown contours, cusps, gingival margin) is an individual
"tooth print". The query is given its best **rigid** fit to every gallery arch
(:func:`align_rigid`) and the arch with the smallest mean surface distance is the
identity (:func:`identify_surface`) — a genuine re-scan fits to ~scan noise, an
impostor's anatomy cannot. The fit is shape, not pose, so the score is a fair
biometric. Feature-based global registration (FPFH/FGR) was evaluated and rejected
here: the self-similar palate and teeth make those descriptors ambiguous.
"""

from __future__ import annotations

import numpy as np


def align_rigid(
    query_points: np.ndarray, gallery_points: np.ndarray, voxel_size: float = 0.5
):
    """Robust **rigid** best-fit of a query scan onto a gallery scan.

    PCA principal-axis initialisation — the four proper-rotation sign hypotheses of
    the arch's principal axes, a *global* init driven by the whole shape (so it needs
    no lucky starting pose, and — unlike feature-based global registration like FGR —
    it is not fooled by the self-similar palate and teeth) — each refined by
    multi-scale Generalized-ICP (Segal et al. 2009, plane-to-plane, the most accurate
    ICP variant), keeping the alignment with the smallest mean surface distance.
    Deterministic (no RANSAC randomness); rigid (no scale, so the query can't collapse
    onto an arbitrary cloud). The residual is therefore a *fair* biometric: ~scan
    noise for a genuine re-scan, the morphological gap for an impostor — and an
    impostor that reads high does so because the shape differs, not the pose.

    Returns ``(aligned_query_points, mean_surface_distance_mm)``.
    """
    import open3d as o3d

    q = np.asarray(query_points, float)
    g = np.asarray(gallery_points, float)
    if q.ndim != 2 or q.shape[1] != 3 or g.ndim != 2 or g.shape[1] != 3:
        raise ValueError("query and gallery points must be (N, 3) arrays")
    reg = o3d.pipelines.registration
    qc, gc = q.mean(0), g.mean(0)
    _, _, Vq = np.linalg.svd(q - qc, full_matrices=False)
    _, _, Vg = np.linalg.svd(g - gc, full_matrices=False)
    dG, dQ = np.linalg.det(Vg), np.linalg.det(Vq)
    qo = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q))
    qo.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30)
    )
    go = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(g))
    go.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30)
    )
    best_md, best_T = np.inf, np.eye(4)
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        sz = sx * sy * dG * dQ  # forces a proper rotation (det +1)
        Rr = Vg.T @ np.diag([sx, sy, sz]) @ Vq
        T = np.eye(4)
        T[:3, :3] = Rr
        T[:3, 3] = gc - Rr @ qc
        for thr in (voxel_size * 4, voxel_size * 2, voxel_size):
            T = reg.registration_generalized_icp(
                qo,
                go,
                thr,
                T,
                reg.TransformationEstimationForGeneralizedICP(),
                reg.ICPConvergenceCriteria(max_iteration=60),
            ).transformation
        al = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q)).transform(T)
        md = float(np.asarray(al.compute_point_cloud_distance(go)).mean())
        if md < best_md:
            best_md, best_T = md, T
    aligned = np.asarray(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q)).transform(best_T).points
    )
    return aligned, best_md


def identify_surface(
    query_points: np.ndarray, gallery_point_sets, voxel_size: float = 0.5
) -> np.ndarray:
    """Identify one query scan against a gallery by best-rigid-fit mean surface distance.

    ``gallery_point_sets`` is a sequence of ``(N, 3)`` arrays. Each candidate is given
    a strong alignment first (:func:`align_rigid`), so the returned per-gallery distance
    row reflects shape, not pose; the argmin is the identity. For the sharpest score use
    :func:`score_to_surface` when the gallery is available as a mesh.
    """
    return np.array(
        [
            align_rigid(query_points, np.asarray(g), voxel_size)[1]
            for g in gallery_point_sets
        ]
    )


def score_to_surface(
    query_points: np.ndarray,
    gallery_vertices: np.ndarray,
    gallery_faces: np.ndarray,
    voxel_size: float = 0.5,
) -> float:
    """Best rigid fit, then mean distance from the query to the gallery *surface* (mm).

    The nearest-gallery-*point* distance used by :func:`identify_surface` has a floor at
    ~half the gallery's point spacing — it can never read zero even for a perfect match.
    Measuring against the gallery *triangle surface* (raycast) removes that floor: a
    genuine re-scan reads ~scan noise (sub-0.1 mm), an impostor stays millimetres off,
    so the genuine/impostor separation is far sharper. Needs the gallery as a mesh
    (vertices + faces, e.g. from :func:`toothprint.io.load_scan`).
    """
    import open3d as o3d

    aligned, _ = align_rigid(query_points, gallery_vertices, voxel_size)
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(gallery_vertices, np.float32)),
        o3d.core.Tensor(np.asarray(gallery_faces, np.uint32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    d = scene.compute_distance(o3d.core.Tensor(np.asarray(aligned, np.float32))).numpy()
    return float(d.mean())
