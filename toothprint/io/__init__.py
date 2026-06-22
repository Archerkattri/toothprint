"""Safe medical-format I/O — every format a dentist or oral radiologist uses.

    from toothprint.io import load
    obj = load("scan.stl")        # -> Scan         (STL/PLY/OBJ/OFF/GLB/3MF)
    obj = load("xray.dcm")        # -> Radiograph   (DICOM, PNG/JPG/TIFF/BMP/...)
    vol = load("cbct.nii.gz")     # -> Volume        (NIfTI; or load_dicom_series(dir))

Detection is by file *content* (magic bytes), not the extension. Every loader is
guarded against the standard medical-parser attacks (oversize files, decompression
bombs, billion-element headers, external-reference smuggling) and raises a
``ValueError`` subclass on anything hostile or corrupt — never an uncaught crash.
The heavy parsers (pydicom, nibabel, trimesh, tifffile) are optional extras imported
lazily, so the certification core stays dependency-light.
"""

from __future__ import annotations

from pathlib import Path

from ._limits import CorruptFile, FileTooLarge, IOError_, UnsupportedFormat
from .detect import RADIOGRAPH, SCAN, VOLUME, detect
from .radiograph import load_radiograph
from .scan import load_scan
from .types import Radiograph, Scan, Volume
from .volume import load_dicom_series, load_volume

SUPPORTED = {
    "radiograph": ["dicom (.dcm)", "png", "jpeg", "tiff", "bmp", "gif", "webp"],
    "scan": ["stl", "ply", "obj", "off", "glb/gltf", "3mf"],
    "volume": ["nifti (.nii/.nii.gz)", "dicom series (directory)"],
}


def load(path):
    """Auto-detect and load any supported file (or DICOM directory) into a
    :class:`Radiograph`, :class:`Scan`, or :class:`Volume`."""
    p = Path(path)
    if p.is_dir():
        return load_dicom_series(p)
    fmt, cat = detect(p)
    if cat == RADIOGRAPH:
        return load_radiograph(p, fmt)
    if cat == SCAN:
        return load_scan(p, fmt)
    if fmt == "nifti":
        return load_volume(p)
    return load_radiograph(p, "dicom")  # single-file DICOM -> first frame


__all__ = [
    "load",
    "load_radiograph",
    "load_scan",
    "load_volume",
    "load_dicom_series",
    "detect",
    "Radiograph",
    "Scan",
    "Volume",
    "SUPPORTED",
    "IOError_",
    "UnsupportedFormat",
    "CorruptFile",
    "FileTooLarge",
    "RADIOGRAPH",
    "SCAN",
    "VOLUME",
]
