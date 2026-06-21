"""Load 3D volumes — NIfTI (.nii/.nii.gz) and DICOM CBCT series (a directory of
slices) — into a normalized Volume with physical spacing in mm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ._limits import (CorruptFile, FileTooLarge, UnsupportedFormat, check_file_size,
                      guard_decompressed, guard_volume, gzip_member_size)
from .types import Volume


def _load_nifti(path: Path) -> Volume:
    import nibabel as nib
    if path.name.lower().endswith(".gz"):
        guard_decompressed(gzip_member_size(path))
    try:
        img = nib.load(str(path))
        shape = img.shape
        guard_volume(int(np.prod(shape[:3])))            # from header, before loading
        arr = np.asarray(img.dataobj, dtype=np.float32)
    except FileTooLarge:
        raise
    except Exception as e:
        raise CorruptFile(f"unreadable NIfTI: {e}")
    zooms = img.header.get_zooms()[:3]
    spacing = tuple(float(z) for z in zooms) if len(zooms) == 3 else (1.0, 1.0, 1.0)
    return Volume(voxels=arr, spacing_mm=spacing, source_format="nifti",
                  meta={"affine": np.asarray(img.affine).tolist()})


def load_volume(path) -> Volume:
    """Load a NIfTI volume file. (For a CBCT *directory* use :func:`load_dicom_series`.)"""
    p = Path(path)
    check_file_size(p)
    if not (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz")):
        raise UnsupportedFormat(f"{p.name} is not a NIfTI volume")
    return _load_nifti(p)


def load_dicom_series(directory) -> Volume:
    """Load a CBCT / CT volume from a directory of DICOM slices, sorted into order."""
    import pydicom
    d = Path(directory)
    if not d.is_dir():
        raise UnsupportedFormat(f"{d} is not a directory of DICOM slices")
    slices = []
    total = 0
    for f in sorted(d.iterdir()):
        if not f.is_file():
            continue
        total += f.stat().st_size
        guard_decompressed(total)
        try:
            ds = pydicom.dcmread(str(f), force=False)
        except Exception:
            continue                                      # skip non-DICOM files in the dir
        if "PixelData" in ds:
            slices.append(ds)
    if not slices:
        raise CorruptFile("no readable DICOM slices in directory")
    rows = int(getattr(slices[0], "Rows", 0)); cols = int(getattr(slices[0], "Columns", 0))
    guard_volume(rows * cols * len(slices))

    def _z(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            return float(ipp[2])
        return float(getattr(ds, "InstanceNumber", 0) or 0)   # pragma: no cover - position fallback

    slices.sort(key=_z)
    try:
        vol = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)
    except Exception as e:                               # pragma: no cover - malformed series codec
        raise CorruptFile(f"cannot decode DICOM series pixels ({e})")
    ps = getattr(slices[0], "PixelSpacing", [1.0, 1.0])
    st = float(getattr(slices[0], "SliceThickness", 1.0) or 1.0)
    spacing = (st, float(ps[0]), float(ps[1]))
    return Volume(voxels=vol, spacing_mm=spacing, source_format="dicom_series",
                  meta={"n_slices": len(slices), "modality": str(getattr(slices[0], "Modality", "")) or None})
