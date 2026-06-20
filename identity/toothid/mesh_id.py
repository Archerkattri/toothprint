"""3D dental biometric identification from intraoral-scan point clouds.

Implements the registration-based identification pipeline that is SOTA for dental
biometrics (Bioengineering 2024; >96-100% Rank-1): preprocess -> FPFH features ->
coarse RANSAC alignment -> fine ICP -> the gallery model with the smallest
registration RMSE is the identity. Genuine matches register to a low RMSE; an
impostor (different person's teeth) cannot, so RMSE separates them cleanly.

A person is recognised by their teeth: the unique 3D shape of the dental arch
(crown contours, cusps, gingival margin) acts as a biometric "tooth print".
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Point-cloud preprocessing + features (open3d)
# ---------------------------------------------------------------------------

def to_point_cloud(points: np.ndarray, voxel_size: float):
    """Voxel-downsample points and estimate normals -> open3d PointCloud."""
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


def register_rmse(query_pcd, query_fpfh, gallery_pcd, gallery_fpfh, voxel_size: float):
    """Coarse RANSAC (FPFH) + fine ICP; return (inlier_rmse, fitness).

    A lower RMSE with a higher fitness means the two arches are the same surface.
    """
    import open3d as o3d

    distance = voxel_size * 1.5
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        query_pcd, gallery_pcd, query_fpfh, gallery_fpfh, True, distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    fine = o3d.pipelines.registration.registration_icp(
        query_pcd, gallery_pcd, distance, coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return float(fine.inlier_rmse), float(fine.fitness)


def enroll(points: np.ndarray, voxel_size: float):
    """Preprocess + featurize one subject's scan into a gallery/query template."""
    pcd = to_point_cloud(points, voxel_size)
    return pcd, compute_fpfh(pcd, voxel_size)


# ---------------------------------------------------------------------------
# Identification logic (pure; no open3d) — testable in isolation
# ---------------------------------------------------------------------------

def rank1_match(rmse_row: np.ndarray) -> int:
    """Index of the gallery model with the smallest registration RMSE."""
    r = np.asarray(rmse_row, dtype=float)
    if r.size == 0:
        raise ValueError("rmse_row must be non-empty")
    return int(np.argmin(r))


def identification_metrics(rmse: np.ndarray, query_labels, gallery_labels) -> dict:
    """Rank-1 accuracy + genuine/impostor RMSE separation from an RMSE matrix.

    ``rmse[i, j]`` is the registration RMSE of query i against gallery j.
    Returns rank-1 accuracy, the genuine/impostor RMSE distributions, and a
    decidability score d' = |mu_g - mu_i| / sqrt((var_g + var_i)/2).
    """
    rmse = np.asarray(rmse, dtype=float)
    q = list(query_labels)
    g = list(gallery_labels)
    if rmse.shape != (len(q), len(g)):
        raise ValueError("rmse shape must be (n_query, n_gallery)")
    correct = 0
    genuine, impostor = [], []
    for i in range(len(q)):
        j = rank1_match(rmse[i])
        if g[j] == q[i]:
            correct += 1
        for jj in range(len(g)):
            (genuine if g[jj] == q[i] else impostor).append(rmse[i, jj])
    genuine = np.asarray(genuine)
    impostor = np.asarray(impostor)
    dprime = 0.0
    if genuine.size and impostor.size:
        denom = np.sqrt((genuine.var() + impostor.var()) / 2.0)
        dprime = float(abs(genuine.mean() - impostor.mean()) / denom) if denom > 0 else 0.0
    return {
        "rank1_accuracy": correct / len(q),
        "n_query": len(q),
        "n_gallery": len(g),
        "genuine_rmse_mean": float(genuine.mean()) if genuine.size else float("nan"),
        "impostor_rmse_mean": float(impostor.mean()) if impostor.size else float("nan"),
        "genuine_rmse_max": float(genuine.max()) if genuine.size else float("nan"),
        "impostor_rmse_min": float(impostor.min()) if impostor.size else float("nan"),
        "decidability_dprime": dprime,
    }
