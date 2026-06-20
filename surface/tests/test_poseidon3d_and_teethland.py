"""Tests for Poseidon3DLoader and TeethLandLoader."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from dentalmapcert.dataset_loaders import (
    Poseidon3DLoader,
    TeethLandLoader,
    load_poseidon3d_points,
    load_teethland_points,
    registry,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
POSEIDON_ROOT = os.environ.get(
    "DMC_POSEIDON_ROOT", str(_REPO_ROOT / "data/poseidon3d/extracted/data")
)
TEETHLAND_ROOT = os.environ.get(
    "DMC_TEETHLAND_ROOT", str(_REPO_ROOT / "data/teeth3ds/extracted")
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def poseidon_root():
    if not Path(POSEIDON_ROOT).exists():
        pytest.skip("Poseidon3D data not found")
    return POSEIDON_ROOT


@pytest.fixture
def teethland_root():
    if not Path(TEETHLAND_ROOT).exists():
        pytest.skip("3DTeethLand data not found")
    return TEETHLAND_ROOT


# ---------------------------------------------------------------------------
# Poseidon3D — nonexistent root
# ---------------------------------------------------------------------------


def test_poseidon_nonexistent_root_empty_records(tmp_path):
    """Nonexistent root should yield zero records."""
    loader = Poseidon3DLoader(str(tmp_path / "nonexistent"))
    assert list(loader.records()) == []


def test_poseidon_nonexistent_root_validate_errors(tmp_path):
    """validate_paths should report errors for a nonexistent root."""
    loader = Poseidon3DLoader(str(tmp_path / "nonexistent"))
    errors = loader.validate_paths()
    assert len(errors) > 0


# ---------------------------------------------------------------------------
# Poseidon3D — real data
# ---------------------------------------------------------------------------


def test_poseidon_records_count(poseidon_root):
    """Should yield well over 100 records (one per arch per case, ~200 cases)."""
    loader = Poseidon3DLoader(poseidon_root)
    records = list(loader.records())
    assert len(records) > 100


def test_poseidon_mesh_paths_exist(poseidon_root):
    """Every yielded record's mesh_path must point to a file that exists."""
    loader = Poseidon3DLoader(poseidon_root)
    for rec in loader.records():
        assert rec.mesh_path is not None, f"mesh_path is None for {rec.record_id}"
        assert Path(rec.mesh_path).exists(), f"STL not found: {rec.mesh_path}"


def test_poseidon_tooth_ids_is_list(poseidon_root):
    """tooth_ids_fdi must be a list for every record."""
    loader = Poseidon3DLoader(poseidon_root)
    for rec in loader.records():
        assert isinstance(rec.tooth_ids_fdi, list), (
            f"tooth_ids_fdi is not a list for {rec.record_id}"
        )


def test_poseidon_splits_valid(poseidon_root):
    """All records must have a valid split value."""
    loader = Poseidon3DLoader(poseidon_root)
    valid_splits = {"train", "val", "test"}
    for rec in loader.records():
        assert rec.split in valid_splits, f"Invalid split '{rec.split}' for {rec.record_id}"


# ---------------------------------------------------------------------------
# Poseidon3D — load_poseidon3d_points
# ---------------------------------------------------------------------------


def test_load_poseidon3d_points(poseidon_root):
    """load_poseidon3d_points should return (n_points, 3) if open3d is available."""
    pytest.importorskip("open3d", reason="open3d not installed")
    loader = Poseidon3DLoader(poseidon_root)
    rec = next(iter(loader.records()))
    n_points = 1000
    pts = load_poseidon3d_points(rec, n_points=n_points)
    assert isinstance(pts, np.ndarray)
    assert pts.shape == (n_points, 3), f"Expected ({n_points}, 3), got {pts.shape}"


def test_load_poseidon3d_points_missing_file_raises(tmp_path):
    """A missing mesh file fast-fails with FileNotFoundError."""
    from dentalmapcert.dataset_loaders import DatasetRecord

    rec = DatasetRecord(
        record_id="dummy",
        dataset_name="poseidon3d",
        image_path=None,
        mesh_path=str(tmp_path / "dummy.stl"),  # does not exist
        label_path=None,
        split="train",
    )
    with pytest.raises(FileNotFoundError, match="STL not found"):
        load_poseidon3d_points(rec)


def test_load_poseidon3d_points_none_mesh_path_raises():
    """A None mesh_path fast-fails with FileNotFoundError."""
    from dentalmapcert.dataset_loaders import DatasetRecord

    rec = DatasetRecord(
        record_id="dummy",
        dataset_name="poseidon3d",
        image_path=None,
        mesh_path=None,
        label_path=None,
        split="train",
    )
    with pytest.raises(FileNotFoundError, match="STL not found"):
        load_poseidon3d_points(rec)


# ---------------------------------------------------------------------------
# 3DTeethLand — nonexistent root
# ---------------------------------------------------------------------------


def test_teethland_nonexistent_root_empty_records(tmp_path):
    """Nonexistent root should yield zero records."""
    loader = TeethLandLoader(str(tmp_path / "nonexistent"))
    assert list(loader.records()) == []


# ---------------------------------------------------------------------------
# 3DTeethLand — real data
# ---------------------------------------------------------------------------


def test_teethland_records_count(teethland_root):
    """Should yield well over 100 records (120 upper + 120 lower = 240)."""
    loader = TeethLandLoader(teethland_root)
    records = list(loader.records())
    assert len(records) > 100


def test_teethland_label_path_set(teethland_root):
    """Every yielded record should have label_path set to an existing JSON file."""
    loader = TeethLandLoader(teethland_root)
    for rec in loader.records():
        assert rec.label_path is not None, f"label_path is None for {rec.record_id}"
        assert Path(rec.label_path).exists(), f"JSON not found: {rec.label_path}"


def test_teethland_splits_valid(teethland_root):
    """All records must have a valid split value."""
    loader = TeethLandLoader(teethland_root)
    valid_splits = {"train", "val", "test"}
    for rec in loader.records():
        assert rec.split in valid_splits, f"Invalid split '{rec.split}' for {rec.record_id}"


# ---------------------------------------------------------------------------
# load_teethland_points
# ---------------------------------------------------------------------------


def test_load_teethland_points_shape(teethland_root):
    """load_teethland_points should return Nx3 array with N > 10 for first record."""
    loader = TeethLandLoader(teethland_root)
    rec = next(iter(loader.records()))
    pts = load_teethland_points(rec)
    assert isinstance(pts, np.ndarray)
    assert pts.ndim == 2
    assert pts.shape[1] == 3
    assert pts.shape[0] > 10, f"Expected more than 10 landmarks, got {pts.shape[0]}"


def test_load_teethland_points_none_label_path():
    """Returns empty (0,3) array when label_path is None."""
    from dentalmapcert.dataset_loaders import DatasetRecord

    rec = DatasetRecord(
        record_id="dummy",
        dataset_name="3dteethland",
        image_path=None,
        mesh_path=None,
        label_path=None,
        split="train",
    )
    pts = load_teethland_points(rec)
    assert pts.shape == (0, 3)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_new_loaders():
    """registry() must include both new loader classes."""
    reg = registry()
    assert "poseidon3d" in reg
    assert "3dteethland" in reg
    assert reg["poseidon3d"] is Poseidon3DLoader
    assert reg["3dteethland"] is TeethLandLoader


# ---------------------------------------------------------------------------
# Poseidon3DLoader.name (line 248)
# ---------------------------------------------------------------------------

def test_poseidon_loader_name(tmp_path):
    """Poseidon3DLoader.name returns 'poseidon3d' (line 248)."""
    loader = Poseidon3DLoader(str(tmp_path))
    assert loader.name == "poseidon3d"


# ---------------------------------------------------------------------------
# Poseidon3D _load_metadata — missing and invalid JSON (lines 253, 259-261)
# ---------------------------------------------------------------------------

def test_poseidon_load_metadata_no_json_returns_empty(tmp_path):
    """_load_metadata returns [] when metadata.json doesn't exist (line 253)."""
    root = tmp_path / "poseidon_no_meta"
    root.mkdir()
    loader = Poseidon3DLoader(str(root))
    result = loader._load_metadata()
    assert result == []


def test_poseidon_load_metadata_invalid_json_returns_empty(tmp_path):
    """_load_metadata returns [] and warns when metadata.json is not valid JSON (lines 259-261)."""
    root = tmp_path / "poseidon_bad_meta"
    root.mkdir()
    (root / "metadata.json").write_text("NOT_JSON{{")
    loader = Poseidon3DLoader(str(root))
    result = loader._load_metadata()
    assert result == []


# ---------------------------------------------------------------------------
# Poseidon3D records() — root exists with valid metadata (line 253 coverage via records())
# ---------------------------------------------------------------------------

def test_poseidon_records_with_empty_metadata_yields_nothing(tmp_path):
    """records() yields 0 records when metadata.json exists but is empty list (line 253 via records)."""
    import json
    root = tmp_path / "poseidon_empty_meta"
    root.mkdir()
    (root / "metadata.json").write_text(json.dumps([]))
    loader = Poseidon3DLoader(str(root))
    records = list(loader.records())
    assert records == []


# ---------------------------------------------------------------------------
# Poseidon3D validate_paths — root exists but no metadata.json (lines 316-319)
# ---------------------------------------------------------------------------

def test_poseidon_validate_paths_root_exists_no_metadata(tmp_path):
    """validate_paths reports metadata.json missing when root exists (lines 316-319)."""
    root = tmp_path / "poseidon_no_meta"
    root.mkdir()
    loader = Poseidon3DLoader(str(root))
    errors = loader.validate_paths()
    assert any("metadata.json" in e for e in errors)


# ---------------------------------------------------------------------------
# Poseidon3D validate_paths — metadata.json exists, missing STL (lines 321-331)
# ---------------------------------------------------------------------------

def test_poseidon_validate_paths_metadata_with_missing_stl(tmp_path):
    """validate_paths reports STL not found when metadata references a non-existent file (lines 321-330)."""
    import json
    root = tmp_path / "poseidon_with_meta"
    root.mkdir()
    meta = [{"id": "case_001", "mandible_paths": ["case_001/mandible.stl"], "maxilla_paths": []}]
    (root / "metadata.json").write_text(json.dumps(meta))
    loader = Poseidon3DLoader(str(root))
    errors = loader.validate_paths()
    assert any("STL not found" in e for e in errors)


# ---------------------------------------------------------------------------
# load_poseidon3d_points — open3d ImportError with existing file (lines 349-351)
# ---------------------------------------------------------------------------

def test_load_poseidon3d_points_open3d_import_error_with_real_file(tmp_path, monkeypatch):
    """load_poseidon3d_points raises RuntimeError when open3d is blocked but mesh file exists."""
    import sys
    from dentalmapcert.dataset_loaders import DatasetRecord

    stl_path = tmp_path / "dummy.stl"
    stl_path.write_bytes(b"solid dummy\nendsolid dummy\n")

    monkeypatch.setitem(sys.modules, "open3d", None)
    rec = DatasetRecord(
        record_id="dummy",
        dataset_name="poseidon3d",
        image_path=None,
        mesh_path=str(stl_path),
        label_path=None,
        split="train",
    )
    with pytest.raises(RuntimeError, match="open3d is required"):
        load_poseidon3d_points(rec)


def test_load_poseidon3d_points_oserror_wrapped_in_runtime_error(tmp_path):
    """An OSError from Open3D reading the file is wrapped in a RuntimeError."""
    pytest.importorskip("open3d", reason="open3d not installed")
    from unittest.mock import patch
    from dentalmapcert.dataset_loaders import DatasetRecord
    import open3d as o3d

    stl_path = tmp_path / "mesh.stl"
    stl_path.write_bytes(b"solid dummy\nendsolid dummy\n")
    rec = DatasetRecord(
        record_id="dummy",
        dataset_name="poseidon3d",
        image_path=None,
        mesh_path=str(stl_path),
        label_path=None,
        split="train",
    )
    with patch.object(o3d.io, "read_triangle_mesh", side_effect=OSError("cannot read")):
        with pytest.raises(RuntimeError, match="Failed to read STL"):
            load_poseidon3d_points(rec)


def test_load_poseidon3d_points_runtime_error_propagates(tmp_path):
    """A RuntimeError during reading propagates instead of being swallowed."""
    pytest.importorskip("open3d", reason="open3d not installed")
    from unittest.mock import patch
    from dentalmapcert.dataset_loaders import DatasetRecord
    import open3d as o3d

    stl_path = tmp_path / "mesh.stl"
    stl_path.write_bytes(b"solid dummy\nendsolid dummy\n")
    rec = DatasetRecord(
        record_id="dummy", dataset_name="poseidon3d", image_path=None,
        mesh_path=str(stl_path), label_path=None, split="train",
    )
    with patch.object(o3d.io, "read_triangle_mesh", side_effect=RuntimeError("read failed")):
        with pytest.raises(RuntimeError, match="read failed"):
            load_poseidon3d_points(rec)


def test_load_poseidon3d_points_empty_mesh_raises(tmp_path):
    """An STL that parses to a mesh with no triangles fast-fails with ValueError."""
    pytest.importorskip("open3d", reason="open3d not installed")
    from unittest.mock import patch
    from dentalmapcert.dataset_loaders import DatasetRecord
    import open3d as o3d

    stl_path = tmp_path / "mesh.stl"
    stl_path.write_bytes(b"solid dummy\nendsolid dummy\n")
    rec = DatasetRecord(
        record_id="dummy", dataset_name="poseidon3d", image_path=None,
        mesh_path=str(stl_path), label_path=None, split="train",
    )
    empty_mesh = o3d.geometry.TriangleMesh()  # no vertices/triangles
    with patch.object(o3d.io, "read_triangle_mesh", return_value=empty_mesh):
        with pytest.raises(ValueError, match="no triangles"):
            load_poseidon3d_points(rec)


def test_load_poseidon3d_points_seed_is_reproducible(poseidon_root):
    """Seeding Open3D's RNG makes the uniform surface sampling reproducible."""
    loader = Poseidon3DLoader(str(poseidon_root))
    records = list(loader.records())
    if not records or records[0].mesh_path is None:
        import pytest
        pytest.skip("no Poseidon3D mesh available")
    a = load_poseidon3d_points(records[0], n_points=500, seed=7)
    b = load_poseidon3d_points(records[0], n_points=500, seed=7)
    assert a.shape == b.shape
    if a.shape[0] > 0:
        import numpy as np
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# TeethLandLoader.name (line 369)
# ---------------------------------------------------------------------------

def test_teethland_loader_name(tmp_path):
    """TeethLandLoader.name returns '3dteethland' (line 369)."""
    loader = TeethLandLoader(str(tmp_path))
    assert loader.name == "3dteethland"


# ---------------------------------------------------------------------------
# TeethLand records() — arch_dir missing → continue (line 382)
# ---------------------------------------------------------------------------

def test_teethland_records_arch_dir_missing_is_skipped(tmp_path):
    """records() skips arch dir if it doesn't exist (line 382 continue)."""
    root = tmp_path / "teethland_no_arch"
    root.mkdir()
    # No upper/ or lower/ dirs exist
    loader = TeethLandLoader(str(root))
    records = list(loader.records())
    assert records == []


# ---------------------------------------------------------------------------
# TeethLand records() — non-dir case_dir → continue (line 385)
# ---------------------------------------------------------------------------

def test_teethland_records_non_dir_case_entry_skipped(tmp_path):
    """records() skips non-directory entries inside arch dir (line 385 continue)."""
    root = tmp_path / "teethland_file_in_arch"
    (root / "upper").mkdir(parents=True)
    (root / "upper" / "README.txt").write_text("not a case")
    loader = TeethLandLoader(str(root))
    records = list(loader.records())
    assert records == []


# ---------------------------------------------------------------------------
# TeethLand records() — fallback to glob when named JSON missing (lines 391-394)
# ---------------------------------------------------------------------------

def test_teethland_records_fallback_json_glob_found(tmp_path):
    """records() uses the first glob-found JSON when named file is missing (lines 391-394)."""
    root = tmp_path / "teethland_glob"
    case_dir = root / "upper" / "case_001"
    case_dir.mkdir(parents=True)
    # Write a JSON with a different name (not the expected pattern)
    (case_dir / "alternative_kpt.json").write_text('{"objects": []}')
    loader = TeethLandLoader(str(root))
    records = list(loader.records())
    # Should yield one record using the fallback json
    assert len(records) == 1
    assert "case_001" in records[0].record_id


def test_teethland_records_no_json_at_all_skips_case(tmp_path):
    """records() skips a case_dir when no JSON file exists at all (lines 392-393 continue)."""
    root = tmp_path / "teethland_no_json"
    case_dir = root / "upper" / "case_002"
    case_dir.mkdir(parents=True)
    # No JSON files — only a txt file
    (case_dir / "notes.txt").write_text("no json here")
    loader = TeethLandLoader(str(root))
    records = list(loader.records())
    assert records == []


# ---------------------------------------------------------------------------
# TeethLandLoader.validate_paths (lines 409-417)
# ---------------------------------------------------------------------------

def test_teethland_validate_paths_nonexistent_root(tmp_path):
    """validate_paths reports missing root (lines 409-412)."""
    loader = TeethLandLoader(str(tmp_path / "nonexistent"))
    errors = loader.validate_paths()
    assert len(errors) > 0
    assert any("does not exist" in e for e in errors)


def test_teethland_validate_paths_root_exists_missing_dirs(tmp_path):
    """validate_paths reports missing upper/ and lower/ dirs (lines 413-417)."""
    root = tmp_path / "teethland_no_dirs"
    root.mkdir()
    loader = TeethLandLoader(str(root))
    errors = loader.validate_paths()
    missing = {e for e in errors if "missing" in e.lower()}
    assert any("upper" in e for e in missing)
    assert any("lower" in e for e in missing)


# ---------------------------------------------------------------------------
# load_teethland_points — empty coords and exception (lines 436, 438-440)
# ---------------------------------------------------------------------------

def test_load_teethland_points_empty_coords_returns_empty(tmp_path):
    """load_teethland_points returns empty when JSON has no 'coord' objects (line 436)."""
    import json
    from dentalmapcert.dataset_loaders import DatasetRecord

    json_path = tmp_path / "landmarks.json"
    json_path.write_text(json.dumps({"objects": [{"label": "tooth", "no_coord_here": []}]}))
    rec = DatasetRecord(
        record_id="r",
        dataset_name="3dteethland",
        image_path=None,
        mesh_path=None,
        label_path=str(json_path),
        split="train",
    )
    pts = load_teethland_points(rec)
    assert pts.shape == (0, 3)


def test_load_teethland_points_invalid_json_returns_empty(tmp_path):
    """load_teethland_points returns empty when JSON is malformed (lines 438-440)."""
    from dentalmapcert.dataset_loaders import DatasetRecord

    json_path = tmp_path / "bad.json"
    json_path.write_text("NOT_VALID_JSON{{{{")
    rec = DatasetRecord(
        record_id="r",
        dataset_name="3dteethland",
        image_path=None,
        mesh_path=None,
        label_path=str(json_path),
        split="train",
    )
    pts = load_teethland_points(rec)
    assert pts.shape == (0, 3)
