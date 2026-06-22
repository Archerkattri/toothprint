"""Load 3D scans (STL, PLY, OBJ, OFF, GLB, 3MF) into a normalized Scan, in mm.

Safety: materials/textures are skipped (an OBJ ``mtllib`` or a GLB URI is an external
reference an attacker could point at an arbitrary path), zip-based 3MF is checked for
a decompression bomb before parsing, and element counts are capped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._limits import CorruptFile, UnsupportedFormat, guard_decompressed, guard_mesh
from .types import Scan


def _zip_bomb_guard(path: Path) -> None:
    import zipfile

    try:
        with zipfile.ZipFile(path) as z:
            total = sum(zi.file_size for zi in z.infolist())
    except Exception as e:
        raise CorruptFile(f"bad zip container: {e}")
    guard_decompressed(total)


def load_scan(path, fmt: str | None = None) -> Scan:
    """Load any 3D surface scan into a :class:`Scan`. Raises an IOError_ subclass."""
    p = Path(path)
    if fmt is None:
        from .detect import detect, SCAN

        fmt, cat = detect(p)
        if cat != SCAN:
            raise UnsupportedFormat(f"{p.name} is a {cat}, not a 3D scan")
    if fmt == "3mf":
        _zip_bomb_guard(p)

    import trimesh

    try:
        geom = trimesh.load(str(p), process=False, skip_materials=True)
    except Exception as e:
        raise CorruptFile(f"unreadable scan ({e})")
    if isinstance(geom, trimesh.Scene):
        if not geom.geometry:  # pragma: no cover - empty scene
            raise CorruptFile("scan contains no geometry")
        geom = trimesh.util.concatenate(list(geom.geometry.values()))

    vertices = np.asarray(getattr(geom, "vertices", np.empty((0, 3))), dtype=np.float64)
    if (
        vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0
    ):  # pragma: no cover - degenerate geometry
        raise CorruptFile(f"scan has no usable vertices (shape {vertices.shape})")
    faces_attr = getattr(geom, "faces", None)
    faces = (
        np.asarray(faces_attr, dtype=np.int64)
        if faces_attr is not None and len(faces_attr)
        else None
    )
    guard_mesh(len(vertices), 0 if faces is None else len(faces))
    if not np.isfinite(vertices).all():
        raise CorruptFile("scan has non-finite vertex coordinates")
    return Scan(
        vertices=vertices,
        faces=faces,
        source_format=fmt,
        meta={"watertight": bool(getattr(geom, "is_watertight", False))},
    )
