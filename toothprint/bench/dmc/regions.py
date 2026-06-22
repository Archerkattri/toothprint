"""Dental surface region taxonomy for DentalMapCert.

Maps FDI tooth numbers to standardized surface region identifiers.
Connects to vertex_indices_path files (.npy) already referenced in
CertificateInput schemas.

FDI notation:
  Upper right:  11-18  Lower left:  31-38
  Upper left:   21-28  Lower right: 41-48
  (Deciduous: 51-55, 61-65, 71-75, 81-85)
"""

from __future__ import annotations
import numpy as np
from pathlib import Path

# Surface regions per tooth
TOOTH_SURFACES = ("buccal", "lingual", "mesial", "distal", "occlusal", "palatal")

# Permanent FDI tooth numbers
PERMANENT_UPPER = tuple(range(11, 19)) + tuple(range(21, 29))  # 11-18, 21-28
PERMANENT_LOWER = tuple(range(31, 39)) + tuple(range(41, 49))  # 31-38, 41-48
PERMANENT_TEETH = PERMANENT_UPPER + PERMANENT_LOWER

# Deciduous FDI tooth numbers
DECIDUOUS_UPPER = tuple(range(51, 56)) + tuple(range(61, 66))
DECIDUOUS_LOWER = tuple(range(71, 76)) + tuple(range(81, 86))
DECIDUOUS_TEETH = DECIDUOUS_UPPER + DECIDUOUS_LOWER

ALL_FDI_TEETH = PERMANENT_TEETH + DECIDUOUS_TEETH


def tooth_arch(fdi_number: int) -> str:
    """Return 'upper' or 'lower' for an FDI tooth number."""
    quadrant = fdi_number // 10
    if quadrant in (1, 2, 5, 6):
        return "upper"
    elif quadrant in (3, 4, 7, 8):
        return "lower"
    raise ValueError(f"Unknown FDI tooth number: {fdi_number}")


def tooth_side(fdi_number: int) -> str:
    """Return 'right' or 'left' for an FDI tooth number.

    Handles both permanent (quadrants 1-4) and deciduous (quadrants 5-8) teeth,
    mirroring tooth_arch. No FDI tooth is "central" — every quadrant lies on one
    side of the midline.
    """
    quadrant = fdi_number // 10
    if quadrant in (1, 4, 5, 8):
        return "right"
    elif quadrant in (2, 3, 6, 7):
        return "left"
    raise ValueError(f"Unknown FDI tooth number: {fdi_number}")


def tooth_type(fdi_number: int) -> str:
    """Return tooth type: incisor, canine, premolar, or molar."""
    tooth_num = fdi_number % 10
    if tooth_num in (1, 2):
        return "incisor"
    elif tooth_num == 3:
        return "canine"
    elif tooth_num in (4, 5):
        return "premolar"
    elif tooth_num in (6, 7, 8):
        return "molar"
    raise ValueError(f"Unknown tooth number within quadrant: {tooth_num}")


def region_id(fdi_number: int, surface: str) -> str:
    """Return a standardized region ID string for (FDI number, surface).

    Example: (16, "buccal") → "tooth_16_buccal"
    """
    surface = surface.lower()
    if surface not in TOOTH_SURFACES:
        raise ValueError(
            f"Unknown surface {surface!r}. Must be one of {TOOTH_SURFACES}"
        )
    return f"tooth_{fdi_number}_{surface}"


def all_region_ids(fdi_number: int) -> list[str]:
    """Return all surface region IDs for a tooth."""
    return [region_id(fdi_number, s) for s in TOOTH_SURFACES]


def load_vertex_indices(path: str | Path) -> np.ndarray:
    """Load vertex indices from a .npy file.

    The .npy file should contain a 1-D array of integer vertex indices
    specifying which mesh vertices belong to a surface region.

    Returns
    -------
    np.ndarray of shape (K,) with dtype int64.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vertex indices file not found: {path}")
    arr = np.load(str(path))
    return arr.astype(np.int64).ravel()


def save_vertex_indices(indices: np.ndarray, path: str | Path) -> Path:
    """Save vertex indices to a .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), np.asarray(indices, dtype=np.int64))
    return path


def region_surface_from_id(region_id_str: str) -> tuple[int, str]:
    """Parse a region ID string back to (fdi_number, surface).

    Inverse of region_id(). Raises ValueError on malformed input.
    """
    parts = region_id_str.split("_")
    if len(parts) != 3 or parts[0] != "tooth":
        raise ValueError(f"Malformed region ID: {region_id_str!r}")
    try:
        fdi = int(parts[1])
    except ValueError:
        raise ValueError(f"Non-integer FDI in region ID: {region_id_str!r}")
    surface = parts[2]
    if surface not in TOOTH_SURFACES:
        raise ValueError(f"Unknown surface in region ID: {region_id_str!r}")
    return fdi, surface
