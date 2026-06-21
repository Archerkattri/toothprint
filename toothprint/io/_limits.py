"""Security limits and guards for untrusted medical files.

Medical file parsers are a real attack surface: DICOM has had many CVEs (malformed
length fields, decompression bombs in JPEG2000/RLE codecs, a 128-byte preamble that
can smuggle a polyglot executable), mesh formats can declare billions of elements,
and zip-based containers (3MF, compressed NIfTI) enable zip bombs. Every loader in
this package routes through these guards: hard caps on file size, decoded pixels,
mesh elements, and volume voxels, plus magic-byte sniffing so a file is parsed by
*what it is*, not the extension an attacker chose.
"""
from __future__ import annotations

import gzip
from pathlib import Path

# --- hard caps (deliberately generous for real clinical data, but finite) -------
MAX_FILE_BYTES = 1_024 ** 3            # 1 GiB: refuse anything larger up front
MAX_IMAGE_PIXELS = 120_000_000         # ~120 MP radiograph (a 12k x 10k panoramic)
MAX_MESH_VERTICES = 25_000_000         # dense IOS arches are < 2 M
MAX_MESH_FACES = 50_000_000
MAX_VOLUME_VOXELS = 1_500_000_000      # ~1150^3 CBCT
MAX_DECOMPRESSED_BYTES = 2 * 1024 ** 3  # zip/codec-bomb ceiling


class IOError_(ValueError):
    """Base: a file we refuse to load (unsupported, corrupt, or over a limit).

    A subclass of ValueError so callers can treat a hostile/garbage upload as a
    clean *rejection* (abstain / "recapture"), never an uncaught crash.
    """


class UnsupportedFormat(IOError_):
    pass


class CorruptFile(IOError_):
    pass


class FileTooLarge(IOError_):
    pass


def check_file_size(path: Path) -> int:
    """Refuse a path that doesn't exist, isn't a regular file, or exceeds the cap.

    Rejecting non-regular files (FIFOs, devices, symlinks to them) before opening
    prevents a hostile path from blocking the reader forever or escaping the cap.
    """
    p = Path(path)
    if not p.exists():
        raise CorruptFile(f"no such file: {p}")
    if not p.is_file():
        raise UnsupportedFormat(f"not a regular file: {p}")
    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        raise FileTooLarge(f"file is {size} bytes (> {MAX_FILE_BYTES} cap)")
    if size == 0:
        raise CorruptFile(f"empty file: {p}")
    return size


def read_magic(path: Path, n: int = 528) -> bytes:
    """First ``n`` bytes (enough to reach the DICOM 'DICM' marker at offset 128)."""
    with open(path, "rb") as fh:
        return fh.read(n)


def gzip_member_size(path: Path) -> int:
    """Uncompressed size advertised in a gzip footer (ISIZE, last 4 bytes mod 2^32).

    Used to reject a compressed-NIfTI bomb before nibabel inflates it. ISIZE is only
    mod 2^32, so it is a *lower-bound* sanity check, not a guarantee — the per-array
    voxel cap is the real backstop.
    """
    with open(path, "rb") as fh:
        fh.seek(-4, 2)
        import struct
        return struct.unpack("<I", fh.read(4))[0]


def guard_pixels(n_pixels: int) -> None:
    if n_pixels > MAX_IMAGE_PIXELS:
        raise FileTooLarge(f"image has {n_pixels} pixels (> {MAX_IMAGE_PIXELS} cap)")


def guard_mesh(n_vertices: int, n_faces: int) -> None:
    if n_vertices > MAX_MESH_VERTICES:
        raise FileTooLarge(f"mesh has {n_vertices} vertices (> {MAX_MESH_VERTICES} cap)")
    if n_faces > MAX_MESH_FACES:
        raise FileTooLarge(f"mesh has {n_faces} faces (> {MAX_MESH_FACES} cap)")


def guard_volume(n_voxels: int) -> None:
    if n_voxels > MAX_VOLUME_VOXELS:
        raise FileTooLarge(f"volume has {n_voxels} voxels (> {MAX_VOLUME_VOXELS} cap)")


def guard_decompressed(total_bytes: int) -> None:
    """Reject a zip/codec bomb whose decompressed size exceeds the cap. Reads the
    module-level constant at call time, so the limit stays runtime-configurable."""
    if total_bytes > MAX_DECOMPRESSED_BYTES:
        raise FileTooLarge(f"input decompresses to {total_bytes} bytes (> cap)")
