"""3D dental identification from intraoral-scan point clouds.

The registration-based pipeline that is state of the art for dental biometrics:
preprocess -> FPFH features -> global alignment (Fast Global Registration) ->
fine refinement (Generalized-ICP) -> the gallery arch the query best-fits, by mean
surface distance, is the identity (:func:`align_rigid` / :func:`identify_surface`).
A person's dental arch (crown contours, cusps, gingival margin) is an individual
"tooth print"; a genuine re-scan best-fits to ~scan noise, an impostor's anatomy
cannot. (The original FPFH+RANSAC+ICP inlier-RMSE path is kept as :func:`identify`.)
"""
from __future__ import annotations

import numpy as np


def to_point_cloud(points: np.ndarray, voxel_size: float):
    """Voxel-downsample a point set and estimate normals -> open3d PointCloud."""
    import open3d as o3d

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2.0, max_nn=30))
    return pcd


def compute_fpfh(pcd, voxel_size: float):
    """Fast Point Feature Histogram (33-dim) descriptors for each point."""
    import open3d as o3d

    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100))


def enroll(points: np.ndarray, voxel_size: float = 0.5):
    """Preprocess + featurize one scan into a (point cloud, features) template."""
    pcd = to_point_cloud(points, voxel_size)
    return pcd, compute_fpfh(pcd, voxel_size)


def register_rmse(query, query_fpfh, gallery, gallery_fpfh, voxel_size: float = 0.5):
    """Coarse RANSAC (FPFH) + fine ICP; return ``(inlier_rmse, fitness)``.

    Lower RMSE with higher fitness means the two arches are the same surface.
    """
    import open3d as o3d

    d = voxel_size * 1.5
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        query, gallery, query_fpfh, gallery_fpfh, True, d,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(d)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    fine = o3d.pipelines.registration.registration_icp(
        query, gallery, d, coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return float(fine.inlier_rmse), float(fine.fitness)


def align_rigid(query_points: np.ndarray, gallery_points: np.ndarray, voxel_size: float = 0.5):
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
    qo.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30))
    go = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(g))
    go.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30))
    best_md, best_T = np.inf, np.eye(4)
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        sz = sx * sy * dG * dQ                          # forces a proper rotation (det +1)
        Rr = Vg.T @ np.diag([sx, sy, sz]) @ Vq
        T = np.eye(4); T[:3, :3] = Rr; T[:3, 3] = gc - Rr @ qc
        for thr in (voxel_size * 4, voxel_size * 2, voxel_size):
            T = reg.registration_generalized_icp(
                qo, go, thr, T, reg.TransformationEstimationForGeneralizedICP(),
                reg.ICPConvergenceCriteria(max_iteration=60)).transformation
        al = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q)).transform(T)
        md = float(np.asarray(al.compute_point_cloud_distance(go)).mean())
        if md < best_md:
            best_md, best_T = md, T
    aligned = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q)).transform(best_T).points)
    return aligned, best_md


def identify_surface(query_points: np.ndarray, gallery_point_sets, voxel_size: float = 0.5) -> np.ndarray:
    """Robust identification: best-rigid-fit mean surface distance of one query scan
    against every gallery scan (``gallery_point_sets`` = sequence of ``(N, 3)``
    arrays). Returns the per-gallery distance row; the argmin is the identity.

    Preferred over :func:`identify` (FPFH + inlier-RMSE), which can be fooled by a
    poor alignment — a strong alignment is given to *every* candidate first via
    :func:`align_rigid`, so the score reflects shape, not pose.
    """
    return np.array([align_rigid(query_points, np.asarray(g), voxel_size)[1]
                     for g in gallery_point_sets])


def identify(query_points: np.ndarray, gallery_templates, voxel_size: float = 0.5,
             no_match_rmse: float = 1e6) -> np.ndarray:
    """RMSE of one query scan against every enrolled gallery template.

    ``gallery_templates`` is a sequence of ``(pcd, fpfh)`` from :func:`enroll`.
    A registration with zero fitness (no usable alignment) scores ``no_match_rmse``.
    Returns the per-gallery RMSE row; the argmin is the identity.
    """
    q_pcd, q_fpfh = enroll(query_points, voxel_size)
    row = np.empty(len(gallery_templates))
    for j, (g_pcd, g_fpfh) in enumerate(gallery_templates):
        rmse, fit = register_rmse(q_pcd, q_fpfh, g_pcd, g_fpfh, voxel_size)
        row[j] = rmse if fit > 0 else no_match_rmse
    return row
