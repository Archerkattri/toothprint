"""Normalized representations every loader returns, so the rest of ToothPrint never
sees a file format — only a radiograph, a scan, or a volume in known units.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Radiograph:
    """A 2D radiograph as float intensity, with MONOCHROME1 already inverted so
    higher always means denser tissue (consistent across DICOM and exported images).
    """
    pixels: np.ndarray                      # 2D float32, intensity (not yet [0,1])
    pixel_spacing_mm: float | None          # in-plane spacing if known, else None
    source_format: str
    modality: str | None = None
    photometric: str | None = None
    bit_depth: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.pixels.shape

    @property
    def normalized(self) -> np.ndarray:
        """Min-max scaled to [0, 1] float32 — what the detectors/registration consume."""
        p = self.pixels.astype(np.float32)
        lo, hi = float(p.min()), float(p.max())
        return (p - lo) / (hi - lo) if hi > lo else np.zeros_like(p)


@dataclass
class Scan:
    """A 3D surface in millimetres — vertices (+ faces if it was a mesh)."""
    vertices: np.ndarray                    # (N, 3) float64, mm
    faces: np.ndarray | None                # (M, 3) int or None for a pure cloud
    source_format: str
    meta: dict = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return 0 if self.faces is None else len(self.faces)

    def to_open3d(self):
        import open3d as o3d
        if self.faces is not None:
            m = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(self.vertices),
                o3d.utility.Vector3iVector(self.faces))
            m.compute_vertex_normals()
            return m
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.vertices))


@dataclass
class Volume:
    """A 3D voxel volume (CBCT / NIfTI) with physical spacing in mm."""
    voxels: np.ndarray                      # 3D float32
    spacing_mm: tuple[float, float, float]  # (z, y, x)
    source_format: str
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.voxels.shape
