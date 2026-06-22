"""Screened-Poisson surface refinement (Kazhdan & Hoppe 2013).

Turns an oriented point cloud into a watertight surface and, as a side effect,
denoises the points. A real quality win in the sub-millimetre regime; past ~2 mm
of noise it over-smooths, so apply it downstream of a good reconstruction.
"""

from __future__ import annotations

import numpy as np


def poisson_refine(
    points: np.ndarray,
    *,
    depth: int = 9,
    density_quantile: float = 0.02,
    normal_radius: float = 2.0,
    n_sample: int = 200_000,
) -> np.ndarray:
    """Refine a point cloud with screened-Poisson reconstruction + resampling.

    Fast-fails (raises) on wrong shape, too few points, missing open3d, or an
    empty Poisson mesh — never silently returns the unrefined input.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")
    if pts.shape[0] < 30:
        raise ValueError("poisson_refine requires at least 30 points")
    if depth < 1:
        raise ValueError("depth must be a positive integer")
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for Poisson refinement") from exc

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(20)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    densities = np.asarray(densities)
    if densities.size and 0.0 < density_quantile < 1.0:
        mesh.remove_vertices_by_mask(
            densities < np.quantile(densities, density_quantile)
        )
    if len(mesh.triangles) == 0:
        raise RuntimeError("Poisson produced an empty mesh")
    refined = mesh.sample_points_uniformly(number_of_points=n_sample)
    return np.asarray(refined.points, dtype=np.float64)
