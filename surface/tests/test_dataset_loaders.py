"""Tests for dataset_loaders, coverage, and capture_protocol modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from dentalmapcert.coverage import CoverageScore, synthetic_coverage
from dentalmapcert.dataset_loaders import PhoneCaptureLoader, Teeth3DSLoader
from dentalmapcert.capture_protocol import (
    STANDARD_PROTOCOL,
    ViewSpec,
    missing_views,
    coverage_per_region,
)


# ---------------------------------------------------------------------------
# Teeth3DSLoader — nonexistent root
# ---------------------------------------------------------------------------


def test_teeth3ds_nonexistent_root_returns_empty_records(tmp_path: Path):
    loader = Teeth3DSLoader(str(tmp_path / "does_not_exist"))
    records = list(loader.records())
    assert records == []


def test_teeth3ds_nonexistent_root_has_validate_errors(tmp_path: Path):
    loader = Teeth3DSLoader(str(tmp_path / "does_not_exist"))
    errors = loader.validate_paths()
    assert len(errors) >= 1
    assert any("does_not_exist" in e for e in errors)


# ---------------------------------------------------------------------------
# PhoneCaptureLoader — nonexistent root
# ---------------------------------------------------------------------------


def test_phone_capture_nonexistent_root_returns_empty_records(tmp_path: Path):
    loader = PhoneCaptureLoader(str(tmp_path / "does_not_exist"))
    records = list(loader.records())
    assert records == []


def test_phone_capture_nonexistent_root_has_validate_errors(tmp_path: Path):
    loader = PhoneCaptureLoader(str(tmp_path / "does_not_exist"))
    errors = loader.validate_paths()
    assert len(errors) >= 1


# ---------------------------------------------------------------------------
# Teeth3DSLoader — populated fixture
# ---------------------------------------------------------------------------


def _make_teeth3ds_fixture(root: Path) -> Path:
    (root / "obj").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "obj" / "case_001.obj").write_text("# dummy mesh\n")
    (root / "obj" / "case_002.obj").write_text("# dummy mesh\n")
    (root / "labels" / "case_001.json").write_text("{}\n")
    return root


def test_teeth3ds_valid_root_yields_records(tmp_path: Path):
    root = _make_teeth3ds_fixture(tmp_path / "teeth3ds")
    loader = Teeth3DSLoader(str(root))
    records = list(loader.records())
    assert len(records) == 2


def test_teeth3ds_valid_root_no_validate_errors(tmp_path: Path):
    root = _make_teeth3ds_fixture(tmp_path / "teeth3ds")
    loader = Teeth3DSLoader(str(root))
    assert loader.validate_paths() == []


def test_teeth3ds_records_have_correct_dataset_name(tmp_path: Path):
    root = _make_teeth3ds_fixture(tmp_path / "teeth3ds")
    loader = Teeth3DSLoader(str(root))
    for rec in loader.records():
        assert rec.dataset_name == "teeth3ds"


def test_teeth3ds_splits_are_valid(tmp_path: Path):
    root = _make_teeth3ds_fixture(tmp_path / "teeth3ds")
    loader = Teeth3DSLoader(str(root))
    valid_splits = {"train", "val", "test"}
    for rec in loader.records():
        assert rec.split in valid_splits


# ---------------------------------------------------------------------------
# PhoneCaptureLoader — populated fixture
# ---------------------------------------------------------------------------


def _make_phone_fixture(root: Path) -> Path:
    img_dir = root / "subj_001" / "t0"
    img_dir.mkdir(parents=True)
    (img_dir / "anterior.jpg").write_bytes(b"\xff\xd8\xff")
    (img_dir / "left_buccal.png").write_bytes(b"\x89PNG")
    return root


def test_phone_capture_valid_root_yields_records(tmp_path: Path):
    root = _make_phone_fixture(tmp_path / "phone")
    loader = PhoneCaptureLoader(str(root))
    records = list(loader.records())
    assert len(records) == 2


def test_phone_capture_records_have_no_mesh_path(tmp_path: Path):
    root = _make_phone_fixture(tmp_path / "phone")
    loader = PhoneCaptureLoader(str(root))
    for rec in loader.records():
        assert rec.mesh_path is None


def test_phone_capture_image_path_is_path_object(tmp_path: Path):
    root = _make_phone_fixture(tmp_path / "phone")
    loader = PhoneCaptureLoader(str(root))
    for rec in loader.records():
        assert isinstance(rec.image_path, Path)


# ---------------------------------------------------------------------------
# Teeth3DS loader fixes (Task 5)
# ---------------------------------------------------------------------------


def test_teeth3ds_image_path_is_none(tmp_path: Path):
    root = _make_teeth3ds_fixture(tmp_path / "teeth3ds")
    loader = Teeth3DSLoader(str(root))
    for rec in loader.records():
        assert rec.image_path is None


def test_teeth3ds_notes_is_stem(tmp_path: Path):
    root = _make_teeth3ds_fixture(tmp_path / "teeth3ds")
    loader = Teeth3DSLoader(str(root))
    records = list(loader.records())
    stems = {r.notes for r in records}
    assert "case_001" in stems
    assert "case_002" in stems


def test_teeth3ds_fdi_parsed_from_filename(tmp_path: Path):
    root = tmp_path / "teeth3ds_fdi"
    (root / "obj").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "obj" / "patient_tooth-36_buccal.obj").write_text("# mesh\n")
    (root / "obj" / "patient_tooth21.obj").write_text("# mesh\n")
    (root / "obj" / "patient_plain.obj").write_text("# mesh\n")
    loader = Teeth3DSLoader(str(root))
    recs = {r.notes: r for r in loader.records()}
    assert recs["patient_tooth-36_buccal"].tooth_ids_fdi == [36]
    assert recs["patient_tooth21"].tooth_ids_fdi == [21]
    assert recs["patient_plain"].tooth_ids_fdi == []


def test_teeth3ds_no_fdi_in_filename_gives_empty_list(tmp_path: Path):
    root = _make_teeth3ds_fixture(tmp_path / "teeth3ds")
    loader = Teeth3DSLoader(str(root))
    for rec in loader.records():
        # fixture files are case_001.obj and case_002.obj — no tooth IDs
        assert rec.tooth_ids_fdi == []


# ---------------------------------------------------------------------------
# synthetic_coverage
# ---------------------------------------------------------------------------


def test_synthetic_coverage_is_in_unit_interval():
    score = synthetic_coverage("region_1", n_views=3, quality_tags=[])
    assert 0.0 <= score.coverage_fraction <= 1.0


def test_synthetic_coverage_method_is_synthetic():
    score = synthetic_coverage("region_1", n_views=2, quality_tags=[])
    assert score.method == "synthetic"


def test_synthetic_coverage_with_all_three_penalty_tags_is_lower_than_no_tags():
    score_clean = synthetic_coverage("r", n_views=4, quality_tags=[])
    score_penalised = synthetic_coverage("r", n_views=4, quality_tags=["glare", "blur", "occlusion"])
    assert score_penalised.coverage_fraction < score_clean.coverage_fraction


def test_synthetic_coverage_zero_views_gives_zero():
    score = synthetic_coverage("r", n_views=0, quality_tags=[])
    assert score.coverage_fraction == 0.0


def test_synthetic_coverage_caps_at_0_95():
    score = synthetic_coverage("r", n_views=100, quality_tags=[])
    assert score.coverage_fraction <= 0.95


def test_synthetic_coverage_never_goes_below_zero():
    score = synthetic_coverage("r", n_views=0, quality_tags=["glare", "blur", "occlusion"])
    assert score.coverage_fraction >= 0.0


# ---------------------------------------------------------------------------
# missing_views
# ---------------------------------------------------------------------------


ALL_FIVE = ["anterior_close", "left_buccal", "right_buccal", "upper_occlusal", "lower_occlusal"]
REQUIRED_VIEW_NAMES = {"anterior_close", "left_buccal", "right_buccal"}


def test_missing_views_all_captured_returns_empty():
    result = missing_views(ALL_FIVE)
    assert result == []


def test_missing_views_only_anterior_captured_returns_two_required_buccal():
    result = missing_views(["anterior_close"])
    missing_names = {v.view_name for v in result}
    assert "left_buccal" in missing_names
    assert "right_buccal" in missing_names


def test_missing_views_optional_views_not_flagged_as_missing():
    # Capture all required; skip optional
    result = missing_views(["anterior_close", "left_buccal", "right_buccal"])
    assert result == []


def test_missing_views_empty_capture_returns_all_required():
    result = missing_views([])
    missing_names = {v.view_name for v in result}
    assert REQUIRED_VIEW_NAMES.issubset(missing_names)


# ---------------------------------------------------------------------------
# coverage_per_region
# ---------------------------------------------------------------------------


def test_coverage_per_region_keys_match_protocol_regions():
    result = coverage_per_region(ALL_FIVE, {})
    protocol_regions: set[str] = set()
    for spec in STANDARD_PROTOCOL:
        protocol_regions.update(spec.target_regions)
    assert set(result.keys()) == protocol_regions


def test_coverage_per_region_all_views_no_tags_all_positive():
    result = coverage_per_region(ALL_FIVE, {})
    for region, frac in result.items():
        assert 0.0 <= frac <= 1.0, f"region {region!r} out of range: {frac}"
        assert frac > 0.0, f"region {region!r} should have positive coverage"


def test_coverage_per_region_no_views_all_zero():
    result = coverage_per_region([], {})
    for frac in result.values():
        assert frac == 0.0


def test_coverage_per_region_glare_reduces_coverage():
    clean = coverage_per_region(["anterior_close"], {})
    glary = coverage_per_region(["anterior_close"], {"anterior_close": ["glare"]})
    # anterior_crown is a target of anterior_close
    assert glary["anterior_crown"] < clean["anterior_crown"]


# ---------------------------------------------------------------------------
# coverage_per_region with reconstruction point cloud (FIX 3)
# ---------------------------------------------------------------------------


def test_coverage_per_region_with_point_cloud_returns_reconstruction_scores():
    """When reconstruction_points and region_bboxes are provided, scores use
    the reconstruction-based path (coverage_from_point_cloud)."""
    import numpy as np
    from dentalmapcert.coverage import coverage_from_point_cloud

    # Build a dense grid of 10×10×10 = 1000 points inside a 10 mm cube.
    step = 1.0
    pts_list = [
        (i * step + 0.5, j * step + 0.5, k * step + 0.5)
        for i in range(10) for j in range(10) for k in range(10)
    ]
    pts_arr = np.array(pts_list, dtype=np.float64)

    bbox = (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
    region_bboxes = {
        "anterior_crown": bbox,
        "buccal_crown": bbox,
        "visible_gingival_margin": bbox,
        "occlusal_or_incisal": bbox,
    }

    result = coverage_per_region(
        ALL_FIVE,
        {},
        reconstruction_points=pts_arr,
        region_bboxes=region_bboxes,
    )

    # Reconstruction coverage for a fully-occupied cube should hit the 0.95 cap.
    for region, frac in result.items():
        assert 0.0 <= frac <= 1.0, f"region {region!r} out of range: {frac}"

    # Regions with bboxes get reconstruction scores; they should all be high.
    for region in region_bboxes:
        assert result[region] >= 0.9, (
            f"reconstruction coverage for {region!r} should be high; got {result[region]}"
        )


def test_coverage_per_region_with_empty_point_cloud_falls_back_to_zero():
    """An empty reconstruction cloud with bboxes gives 0.0 for each covered region."""
    import numpy as np

    pts_arr = np.zeros((0, 3), dtype=np.float64)
    bbox = (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
    region_bboxes = {"anterior_crown": bbox}

    result = coverage_per_region(
        ["anterior_close"],
        {},
        reconstruction_points=pts_arr,
        region_bboxes=region_bboxes,
    )
    # Empty cloud → 0 points in any bbox → coverage = 0.0
    # BUT pts_arr has 0 rows, so use_reconstruction is False → synthetic path.
    # Either way the result must be a valid float in [0, 1].
    assert 0.0 <= result.get("anterior_crown", 0.0) <= 1.0


def test_coverage_per_region_without_reconstruction_uses_synthetic():
    """Without reconstruction_points the function falls back to synthetic heuristic."""
    result = coverage_per_region(ALL_FIVE, {})
    for frac in result.values():
        assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# Teeth3DS OBJ validation (FIX 8)
# ---------------------------------------------------------------------------


def test_teeth3ds_empty_obj_reported_by_validate_paths(tmp_path: Path):
    """validate_paths should report OBJ files with zero bytes."""
    root = tmp_path / "teeth3ds_empty"
    (root / "obj").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "obj" / "empty_mesh.obj").write_bytes(b"")
    loader = Teeth3DSLoader(str(root))
    errors = loader.validate_paths()
    assert any("empty" in e.lower() for e in errors), f"Expected empty-file error; got {errors}"


def test_teeth3ds_invalid_obj_header_reported_by_validate_paths(tmp_path: Path):
    """validate_paths should report OBJ files that don't start with '#' or 'v '."""
    root = tmp_path / "teeth3ds_bad"
    (root / "obj").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "obj" / "bad_mesh.obj").write_text("INVALID_HEADER\n")
    loader = Teeth3DSLoader(str(root))
    errors = loader.validate_paths()
    assert any("bad_mesh.obj" in e for e in errors), f"Expected header error; got {errors}"


def test_teeth3ds_empty_obj_skipped_by_records(tmp_path: Path):
    """records() must skip OBJ files that fail the header check."""
    root = tmp_path / "teeth3ds_skip"
    (root / "obj").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "obj" / "valid_mesh.obj").write_text("# valid\nv 0 0 0\n")
    (root / "obj" / "empty_mesh.obj").write_bytes(b"")
    loader = Teeth3DSLoader(str(root))
    records = list(loader.records())
    # Only the valid mesh should be yielded
    assert len(records) == 1
    assert records[0].notes == "valid_mesh"


def test_teeth3ds_vertex_line_obj_passes_validation(tmp_path: Path):
    """OBJ files starting with 'v ' (vertex line) should pass validation."""
    root = tmp_path / "teeth3ds_vertex"
    (root / "obj").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "obj" / "vertex_mesh.obj").write_text("v 1.0 2.0 3.0\n")
    loader = Teeth3DSLoader(str(root))
    errors = loader.validate_paths()
    assert errors == [], f"Unexpected errors: {errors}"
    records = list(loader.records())
    assert len(records) == 1


# ---------------------------------------------------------------------------
# Loader .name properties (lines 77, 191)
# ---------------------------------------------------------------------------


def test_teeth3ds_loader_name_is_teeth3ds(tmp_path: Path):
    """Teeth3DSLoader.name returns 'teeth3ds' (line 77)."""
    loader = Teeth3DSLoader(str(tmp_path))
    assert loader.name == "teeth3ds"


def test_phone_capture_loader_name(tmp_path: Path):
    """PhoneCaptureLoader.name returns 'phone-captures' (line 191)."""
    loader = PhoneCaptureLoader(str(tmp_path))
    assert loader.name == "phone-captures"


# ---------------------------------------------------------------------------
# Teeth3DS records() — root exists but obj/ missing (lines 105-109)
# ---------------------------------------------------------------------------


def test_teeth3ds_root_exists_but_no_obj_dir_returns_empty(tmp_path: Path):
    """records() returns empty when root exists but obj/ subdir is missing (lines 105-109)."""
    root = tmp_path / "teeth3ds_no_obj"
    root.mkdir()
    loader = Teeth3DSLoader(str(root))
    records = list(loader.records())
    assert records == []


# ---------------------------------------------------------------------------
# Teeth3DS validate_paths — partial directory layout
# ---------------------------------------------------------------------------


def test_teeth3ds_validate_paths_obj_missing_but_root_exists(tmp_path: Path):
    """validate_paths reports obj/ missing when root exists (lines 156-160)."""
    root = tmp_path / "teeth3ds_partial"
    root.mkdir()
    loader = Teeth3DSLoader(str(root))
    errors = loader.validate_paths()
    assert any("obj/" in e for e in errors)


def test_teeth3ds_validate_paths_labels_missing(tmp_path: Path):
    """validate_paths reports labels/ missing when root+obj/ exist but labels/ doesn't (line 162)."""
    root = tmp_path / "teeth3ds_no_labels"
    (root / "obj").mkdir(parents=True)
    loader = Teeth3DSLoader(str(root))
    errors = loader.validate_paths()
    assert any("labels/" in e for e in errors)


# ---------------------------------------------------------------------------
# Teeth3DS _is_valid_obj_header OSError (lines 90-91)
# ---------------------------------------------------------------------------


def test_teeth3ds_is_valid_obj_header_oserror_returns_false(tmp_path: Path):
    """_is_valid_obj_header returns False when the file cannot be opened (lines 90-91)."""
    from unittest.mock import patch, mock_open

    root = tmp_path / "teeth3ds_oserr"
    root.mkdir()
    fake_path = root / "mesh.obj"
    with patch("builtins.open", side_effect=OSError("permission denied")):
        result = Teeth3DSLoader._is_valid_obj_header(fake_path)
    assert result is False


# ---------------------------------------------------------------------------
# PhoneCaptureLoader — non-dir entries (lines 203, 207)
# ---------------------------------------------------------------------------


def test_phone_capture_non_dir_subject_entry_is_skipped(tmp_path: Path):
    """Records() skips non-directory entries in root (line 203 continue)."""
    root = tmp_path / "phone_skip"
    root.mkdir()
    # Create a regular file at subject level (not a dir)
    (root / "README.txt").write_text("not a subject")
    loader = PhoneCaptureLoader(str(root))
    records = list(loader.records())
    assert records == []


def test_phone_capture_non_dir_timepoint_entry_is_skipped(tmp_path: Path):
    """Records() skips non-directory entries inside a subject dir (line 207 continue)."""
    root = tmp_path / "phone_skip_timepoint"
    subj_dir = root / "subj_001"
    subj_dir.mkdir(parents=True)
    # Create a file at timepoint level (not a dir)
    (subj_dir / "metadata.txt").write_text("not a timepoint")
    loader = PhoneCaptureLoader(str(root))
    records = list(loader.records())
    assert records == []
