"""3D dental identification from intraoral-scan point clouds.

The registration-based pipeline that is state of the art for dental biometrics:
preprocess -> FPFH features -> coarse RANSAC alignment -> fine ICP -> the gallery
arch with the smallest registration RMSE is the identity. A person's dental arch
(crown contours, cusps, gingival margin) is an individual "tooth print"; a genuine
re-scan registers to a low RMSE, an impostor's anatomy cannot.
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
