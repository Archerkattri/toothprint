"""Load 2D radiographs (DICOM, PNG/JPG/TIFF/BMP/...) into a normalized Radiograph.

DICOM is handled carefully: rescale slope/intercept (modality LUT), MONOCHROME1
inversion so "higher = denser" is consistent, pixel spacing, and a pixel-count guard
*before* decoding so a malicious header can't trigger a giant allocation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._limits import (
    CorruptFile,
    FileTooLarge,
    MAX_IMAGE_PIXELS,
    UnsupportedFormat,
    guard_pixels,
)
from .types import Radiograph


def _load_dicom(path: Path) -> Radiograph:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    try:
        ds = pydicom.dcmread(str(path), force=False)
    except Exception as e:  # malformed / not DICOM
        raise CorruptFile(f"unreadable DICOM: {e}")
    if "PixelData" not in ds:
        raise CorruptFile("DICOM has no pixel data")
    rows, cols = int(getattr(ds, "Rows", 0)), int(getattr(ds, "Columns", 0))
    frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    guard_pixels(rows * cols * max(frames, 1))  # refuse before decoding
    try:
        arr = np.asarray(ds.pixel_array)  # may invoke a codec
    except Exception as e:  # pragma: no cover - malformed codec stream
        raise CorruptFile(f"cannot decode DICOM pixels ({e})")
    if arr.ndim == 3:
        arr = arr[..., :3].mean(-1) if arr.shape[-1] in (3, 4) else arr[0]
    if arr.ndim != 2:  # pragma: no cover - exotic DICOM shape
        raise CorruptFile(f"unexpected DICOM pixel shape {arr.shape}")
    arr = arr.astype(np.float32)
    try:
        arr = apply_modality_lut(arr, ds).astype(np.float32)
    except Exception:  # pragma: no cover - rescale LUT optional
        pass
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).strip()
    if photometric == "MONOCHROME1":  # 0 = white -> invert
        arr = float(arr.max()) - arr
    spacing = None
    for tag in ("ImagerPixelSpacing", "PixelSpacing"):
        v = getattr(ds, tag, None)
        if v is not None:
            try:
                spacing = float(v[0])
                break
            except Exception:  # pragma: no cover - malformed DS spacing
                pass
    return Radiograph(
        pixels=arr,
        pixel_spacing_mm=spacing,
        source_format="dicom",
        modality=str(getattr(ds, "Modality", "")) or None,
        photometric=photometric or None,
        bit_depth=int(getattr(ds, "BitsStored", 0)) or None,
        meta={"rows": rows, "cols": cols, "frames": frames},
    )


def _load_raster(path: Path, fmt: str) -> Radiograph:
    if fmt == "tiff":
        import tifffile

        try:
            arr = np.asarray(tifffile.imread(str(path)))
        except Exception as e:  # pragma: no cover - malformed TIFF
            raise CorruptFile(f"unreadable TIFF: {e}")
    else:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS  # PIL refuses bombs itself
        try:
            with Image.open(path) as im:
                im.load()
                arr = np.asarray(im)
        except (
            Image.DecompressionBombError
        ) as e:  # pragma: no cover - PIL's own bomb guard (guard_pixels is primary)
            raise FileTooLarge(str(e))
        except Exception as e:
            raise CorruptFile(f"unreadable image: {e}")
    if arr.ndim == 3:
        arr = arr[..., :3].mean(-1)  # luminance
    if arr.ndim != 2:  # pragma: no cover - exotic image shape
        raise CorruptFile(f"unexpected image shape {arr.shape}")
    guard_pixels(int(arr.shape[0]) * int(arr.shape[1]))
    return Radiograph(
        pixels=arr.astype(np.float32),
        pixel_spacing_mm=None,
        source_format=fmt,
        bit_depth=int(arr.dtype.itemsize * 8),
        meta={},
    )


def load_radiograph(path, fmt: str | None = None) -> Radiograph:
    """Load any 2D radiograph into a :class:`Radiograph`. Raises an IOError_ subclass
    (ValueError) for unsupported/corrupt/oversize files — never an uncaught crash."""
    p = Path(path)
    if fmt is None:
        from .detect import detect, RADIOGRAPH

        fmt, cat = detect(p)
        if cat != RADIOGRAPH:
            raise UnsupportedFormat(f"{p.name} is a {cat}, not a radiograph")
    return _load_dicom(p) if fmt == "dicom" else _load_raster(p, fmt)
