"""Dataset manifest loaders for DentalMapCert reference datasets."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Optional

logger = logging.getLogger(__name__)

DatasetSplit = Literal["train", "val", "test"]


@dataclass
class DatasetRecord:
    record_id: str
    dataset_name: str
    image_path: Optional[Path]
    mesh_path: str | None
    label_path: str | None
    split: DatasetSplit
    tooth_ids_fdi: list[int] = field(default_factory=list)
    notes: str = ""


class DatasetLoader(ABC):
    """Abstract base class for dataset manifest loaders."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name."""

    @abstractmethod
    def records(self) -> Iterator[DatasetRecord]:
        """Yield DatasetRecord instances for every sample in the dataset."""

    @abstractmethod
    def validate_paths(self) -> list[str]:
        """Return a list of missing or invalid path error strings.

        An empty list means all expected paths exist.
        """


def _hash_split(key: str, train_frac: float = 0.8, val_frac: float = 0.1) -> DatasetSplit:
    """Deterministic 80/10/10 split based on MD5 hash of a key."""
    digest = int(hashlib.md5(key.encode()).hexdigest(), 16)
    bucket = (digest % 100) / 100.0
    if bucket < train_frac:
        return "train"
    if bucket < train_frac + val_frac:
        return "val"
    return "test"


class Teeth3DSLoader(DatasetLoader):
    """Loader for the Teeth3DS reference dental mesh dataset.

    Expected directory structure::

        root_dir/
          obj/       — .obj mesh files (one per case)
          labels/    — annotation files paired with mesh files
    """

    def __init__(self, root_dir: str) -> None:
        self._root = Path(root_dir)

    @property
    def name(self) -> str:
        return "teeth3ds"

    @staticmethod
    def _is_valid_obj_header(path: Path) -> bool:
        """Return True if *path* starts with a valid OBJ header (comment or vertex line)."""
        try:
            with open(path, "rb") as fh:
                header = fh.read(64)
            if not header:
                return False
            # Decode first line (strip CR/LF)
            first_line = header.split(b"\n")[0].strip()
            return first_line.startswith(b"#") or first_line.startswith(b"v ")
        except OSError:
            return False

    def records(self) -> Iterator[DatasetRecord]:
        obj_dir = self._root / "obj"
        labels_dir = self._root / "labels"

        if not self._root.exists():
            logger.warning(
                "Teeth3DS root directory does not exist: %s — returning empty iterator",
                self._root,
            )
            return

        if not obj_dir.exists():
            logger.warning(
                "Teeth3DS obj/ directory does not exist: %s — returning empty iterator",
                obj_dir,
            )
            return

        for mesh_file in sorted(obj_dir.glob("*.obj")):
            stem = mesh_file.stem

            # Minimal OBJ header check — skip files that look corrupt.
            if not self._is_valid_obj_header(mesh_file):
                logger.warning(
                    "Skipping %s: file is empty or does not start with a valid OBJ header.",
                    mesh_file,
                )
                continue

            record_id = f"teeth3ds_{stem}"
            split = _hash_split(stem)

            # Look for matching label file (any extension under labels/)
            label_path: str | None = None
            if labels_dir.exists():
                candidates = list(labels_dir.glob(f"{stem}.*"))
                if candidates:
                    label_path = str(candidates[0])

            # Parse FDI tooth IDs from filename: matches "tooth-NN", "toothNN", "tooth_NN"
            tooth_ids: list[int] = [
                int(m) for m in re.findall(r"tooth[-_]?(\d{2})", stem, flags=re.IGNORECASE)
            ]

            yield DatasetRecord(
                record_id=record_id,
                dataset_name="teeth3ds",
                image_path=None,  # Teeth3DS is mesh-only; no paired image
                mesh_path=str(mesh_file),
                label_path=label_path,
                split=split,
                tooth_ids_fdi=tooth_ids,
                notes=stem,
            )

    def validate_paths(self) -> list[str]:
        errors: list[str] = []
        if not self._root.exists():
            errors.append(f"root_dir does not exist: {self._root}")
            return errors
        obj_dir = self._root / "obj"
        labels_dir = self._root / "labels"
        if not obj_dir.exists():
            errors.append(
                f"obj/ directory missing: {obj_dir} — Teeth3DS OBJ meshes require "
                "Grand Challenge registration at https://teeth3ds.grand-challenge.org/ "
                "(3DTeethLand landmark JSON files are available separately)"
            )
        if not labels_dir.exists():
            errors.append(f"labels/ directory missing: {labels_dir}")
        # Validate individual OBJ files when obj/ exists.
        if obj_dir.exists():
            for mesh_file in sorted(obj_dir.glob("*.obj")):
                if mesh_file.stat().st_size == 0:
                    errors.append(f"OBJ file is empty: {mesh_file}")
                elif not self._is_valid_obj_header(mesh_file):
                    errors.append(
                        f"OBJ file does not start with '#' comment or 'v ' vertex line: {mesh_file}"
                    )
        return errors


class PhoneCaptureLoader(DatasetLoader):
    """Loader for project-owned phone-capture images.

    Expected directory structure::

        root_dir/
          <subject_id>/
            <timepoint>/
              *.jpg or *.png
    """

    def __init__(self, root_dir: str) -> None:
        self._root = Path(root_dir)

    @property
    def name(self) -> str:
        return "phone-captures"

    def records(self) -> Iterator[DatasetRecord]:
        if not self._root.exists():
            logger.warning(
                "PhoneCapture root directory does not exist: %s — returning empty iterator",
                self._root,
            )
            return

        for subject_dir in sorted(self._root.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject_id = subject_dir.name
            for timepoint_dir in sorted(subject_dir.iterdir()):
                if not timepoint_dir.is_dir():
                    continue
                timepoint = timepoint_dir.name
                image_files = sorted(
                    f
                    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
                    for f in timepoint_dir.glob(ext)
                )
                for image_file in image_files:
                    stem = image_file.stem
                    record_id = f"phonecap_{subject_id}_{timepoint}_{stem}"
                    split = _hash_split(record_id)
                    yield DatasetRecord(
                        record_id=record_id,
                        dataset_name="phone-captures",
                        image_path=image_file,
                        mesh_path=None,
                        label_path=None,
                        split=split,
                        tooth_ids_fdi=[],
                        notes=f"subject={subject_id} timepoint={timepoint}",
                    )

    def validate_paths(self) -> list[str]:
        errors: list[str] = []
        if not self._root.exists():
            errors.append(f"root_dir does not exist: {self._root}")
        return errors


class Poseidon3DLoader(DatasetLoader):
    """Loader for Poseidon3D IOS dental mesh dataset (200 cases, CC-BY-4.0).

    root_dir should be the extracted data/ directory containing metadata.json.
    Yields one DatasetRecord per arch (mandible/maxilla) that has a valid STL.
    """

    def __init__(self, root_dir: str) -> None:
        self._root = Path(root_dir)

    @property
    def name(self) -> str:
        return "poseidon3d"

    def _load_metadata(self) -> list[dict]:
        meta_path = self._root / "metadata.json"
        if not meta_path.exists():
            return []
        try:
            import json

            with open(meta_path) as fh:
                return json.load(fh)
        except Exception:
            logger.warning("Failed to parse Poseidon3D metadata.json: %s", meta_path)
            return []

    def records(self) -> Iterator[DatasetRecord]:
        if not self._root.exists():
            logger.warning(
                "Poseidon3D root directory does not exist: %s — returning empty iterator",
                self._root,
            )
            return

        # metadata.json sits inside root_dir; STL relative paths are relative to
        # root_dir's parent (the "extracted/" directory).
        base = self._root.parent

        for case in self._load_metadata():
            case_id = case.get("id", "")
            for arch in ("mandible", "maxilla"):
                paths_key = f"{arch}_paths"
                rm_key = f"{arch}_region_mapping"
                arch_paths = case.get(paths_key) or []
                if not arch_paths:
                    continue

                rel_stl = arch_paths[0]
                mesh_path = base / rel_stl

                # Look for MARKERS pkl alongside the STL
                markers_name = f"{case_id}_MARKERS_{arch}.pkl"
                markers_path = self._root / case_id / markers_name
                label_path: str | None = str(markers_path) if markers_path.exists() else None

                record_id = f"poseidon3d_{case_id}_{arch}"

                # Extract valid FDI-range tooth IDs from region_mapping
                region_mapping = case.get(rm_key) or []
                tooth_ids_fdi: list[int] = [
                    v for v in region_mapping if isinstance(v, int) and 11 <= v <= 48
                ]

                yield DatasetRecord(
                    record_id=record_id,
                    dataset_name="poseidon3d",
                    image_path=None,
                    mesh_path=str(mesh_path),
                    label_path=label_path,
                    split=_hash_split(record_id),
                    tooth_ids_fdi=tooth_ids_fdi,
                    notes=f"poseidon3d case={case_id} arch={arch}",
                )

    def validate_paths(self) -> list[str]:
        errors: list[str] = []
        if not self._root.exists():
            errors.append(f"root_dir does not exist: {self._root}")
            return errors
        meta_path = self._root / "metadata.json"
        if not meta_path.exists():
            errors.append(f"metadata.json missing: {meta_path}")
            return errors

        base = self._root.parent
        for case in self._load_metadata():
            case_id = case.get("id", "")
            for arch in ("mandible", "maxilla"):
                arch_paths = case.get(f"{arch}_paths") or []
                if not arch_paths:
                    continue
                mesh_path = base / arch_paths[0]
                if not mesh_path.exists():
                    errors.append(f"STL not found: {mesh_path}")
        return errors


def load_poseidon3d_points(
    record: DatasetRecord, n_points: int = 5000, seed: int | None = None
) -> "np.ndarray":
    """Sample n_points from a Poseidon3D STL mesh → Nx3 float64 array.

    Requires open3d. Fast-fails instead of masking problems:

    - ``RuntimeError`` if open3d is not importable.
    - ``FileNotFoundError`` if the STL path is missing/None.
    - ``ValueError`` if the mesh is empty/corrupt (no triangles).
    - Any underlying ``OSError``/``RuntimeError`` from Open3D propagates (the
      OSError is wrapped in a RuntimeError for a clearer message).

    When *seed* is given, Open3D's global RNG is seeded so the uniform surface
    sampling is reproducible (``sample_points_uniformly`` has no seed argument
    and Open3D does not use NumPy's RNG, so this is the only effective seed).
    """
    import numpy as np  # noqa: PLC0415

    if record.mesh_path is None or not Path(record.mesh_path).exists():
        raise FileNotFoundError(f"Poseidon3D STL not found: {record.mesh_path}")
    try:
        import open3d as o3d  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("open3d is required to load Poseidon3D meshes") from exc

    try:
        if seed is not None:
            o3d.utility.random.seed(seed)
        mesh = o3d.io.read_triangle_mesh(str(record.mesh_path))
    except OSError as exc:
        raise RuntimeError(f"Failed to read STL {record.mesh_path}: {exc}") from exc

    if len(mesh.triangles) == 0:
        raise ValueError(f"STL {record.mesh_path} has no triangles (empty/corrupt)")
    pcd = mesh.sample_points_uniformly(n_points)
    return np.asarray(pcd.points, dtype=np.float64)


class TeethLandLoader(DatasetLoader):
    """Loader for 3DTeethLand landmark JSON files (IOS scan landmarks, upper+lower).

    root_dir should be the extracted directory containing upper/ and lower/ subdirectories.
    Yields one DatasetRecord per case+arch.
    """

    def __init__(self, root_dir: str) -> None:
        self._root = Path(root_dir)

    @property
    def name(self) -> str:
        return "3dteethland"

    def records(self) -> Iterator[DatasetRecord]:
        if not self._root.exists():
            logger.warning(
                "3DTeethLand root directory does not exist: %s — returning empty iterator",
                self._root,
            )
            return

        for arch in ("upper", "lower"):
            arch_dir = self._root / arch
            if not arch_dir.exists():
                continue
            for case_dir in sorted(arch_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                case_id = case_dir.name
                json_name = f"{case_id}_{arch}__kpt.json"
                json_path = case_dir / json_name
                if not json_path.exists():
                    # Try any JSON in the directory
                    candidates = list(case_dir.glob("*.json"))
                    if not candidates:
                        continue
                    json_path = candidates[0]

                record_id = f"teethland_{case_id}_{arch}"
                yield DatasetRecord(
                    record_id=record_id,
                    dataset_name="3dteethland",
                    image_path=None,
                    mesh_path=None,
                    label_path=str(json_path),
                    split=_hash_split(record_id),
                    tooth_ids_fdi=[],
                    notes=f"3dteethland case={case_id} arch={arch}",
                )

    def validate_paths(self) -> list[str]:
        errors: list[str] = []
        if not self._root.exists():
            errors.append(f"root_dir does not exist: {self._root}")
            return errors
        for arch in ("upper", "lower"):
            arch_dir = self._root / arch
            if not arch_dir.exists():
                errors.append(f"{arch}/ directory missing: {arch_dir}")
        return errors


def load_teethland_points(record: DatasetRecord) -> "np.ndarray":
    """Load all landmark coordinates from a 3DTeethLand record → Nx3 float64 array.

    Returns empty (0,3) array if label_path is None or JSON cannot be parsed.
    """
    import numpy as np  # noqa: PLC0415

    if record.label_path is None:
        return np.empty((0, 3), dtype=np.float64)
    try:
        import json  # noqa: PLC0415

        with open(record.label_path) as fh:
            data = json.load(fh)
        coords = [obj["coord"] for obj in data.get("objects", []) if "coord" in obj]
        if not coords:
            return np.empty((0, 3), dtype=np.float64)
        return np.array(coords, dtype=np.float64)
    except Exception as exc:
        logger.warning("Failed to parse 3DTeethLand JSON %s: %s", record.label_path, exc)
        return np.empty((0, 3), dtype=np.float64)


def registry() -> dict[str, type[DatasetLoader]]:
    """Return mapping of dataset names to their loader classes."""
    return {
        "teeth3ds": Teeth3DSLoader,
        "phone-captures": PhoneCaptureLoader,
        "poseidon3d": Poseidon3DLoader,
        "3dteethland": TeethLandLoader,
    }
