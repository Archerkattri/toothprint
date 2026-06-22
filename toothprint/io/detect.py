"""Content-based format detection (magic bytes first, extension only as fallback).

An attacker controls the file extension, so we sniff the bytes. Returns a canonical
format id and a category so the dispatcher routes to the right loader.
"""

from __future__ import annotations

from pathlib import Path

from ._limits import UnsupportedFormat, check_file_size, read_magic

# category -> what the bytes become
RADIOGRAPH = "radiograph"  # 2D grayscale image (+ pixel spacing)
SCAN = "scan"  # 3D surface mesh / point cloud (mm)
VOLUME = "volume"  # 3D voxel volume (+ spacing)

_RASTER_EXT = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".bmp": "bmp",
    ".gif": "gif",
    ".webp": "webp",
}
_MESH_EXT = {
    ".stl": "stl",
    ".ply": "ply",
    ".obj": "obj",
    ".off": "off",
    ".glb": "glb",
    ".gltf": "gltf",
    ".3mf": "3mf",
}


def detect(path) -> tuple[str, str]:
    """Return ``(format_id, category)`` for a file. Raises UnsupportedFormat.

    Magic bytes decide whenever a format has them; text/ambiguous mesh formats
    (ASCII STL, OBJ) fall back to the extension.
    """
    p = Path(path)
    check_file_size(p)
    m = read_magic(p)
    ext = p.suffix.lower()
    name = p.name.lower()

    # --- volumes (check before raster: a .nii can start with arbitrary bytes) ---
    if name.endswith(".nii.gz") or name.endswith(".nii"):
        return "nifti", VOLUME
    if len(m) >= 348 and m[344:348] in (b"n+1\x00", b"ni1\x00"):
        return "nifti", VOLUME

    # --- DICOM: 'DICM' marker at byte 128 (preamble), or a .dcm with no preamble ---
    if len(m) >= 132 and m[128:132] == b"DICM":
        return (
            "dicom",
            VOLUME if ext == "" else RADIOGRAPH,
        )  # series-vs-single decided at load
    if ext in (".dcm", ".dicom"):
        return "dicom", RADIOGRAPH

    # --- raster radiographs ---
    if m[:8] == b"\x89PNG\r\n\x1a\n":
        return "png", RADIOGRAPH
    if m[:3] == b"\xff\xd8\xff":
        return "jpeg", RADIOGRAPH
    if m[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "tiff", RADIOGRAPH
    if m[:2] == b"BM":
        return "bmp", RADIOGRAPH
    if m[:6] in (b"GIF87a", b"GIF89a"):
        return "gif", RADIOGRAPH
    if m[:4] == b"RIFF" and m[8:12] == b"WEBP":
        return "webp", RADIOGRAPH

    # --- meshes ---
    if m[:3] == b"ply":
        return "ply", SCAN
    if m[:4] == b"glTF":
        return "glb", SCAN
    if m[:3] == b"OFF":
        return "off", SCAN
    if m[:4] == b"PK\x03\x04" and ext == ".3mf":
        return "3mf", SCAN
    if m[:5].lower() == b"solid" and ext != ".obj":
        return "stl", SCAN  # ASCII STL
    if ext in _MESH_EXT:
        return _MESH_EXT[ext], SCAN  # binary STL / OBJ / etc. by extension
    if ext in _RASTER_EXT:
        return _RASTER_EXT[ext], RADIOGRAPH

    raise UnsupportedFormat(
        f"cannot identify medical format of {p.name} (magic={m[:8]!r}, ext={ext!r})"
    )
