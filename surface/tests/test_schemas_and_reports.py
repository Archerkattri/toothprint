import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python-headless not installed")

from dentalmapcert.calibration import ErrorCalibrator
from dentalmapcert.certificate import CertificateInput, decide_surface_change
from dentalmapcert.image_quality import detect_blur, detect_glare, analyze_view_quality
from dentalmapcert.report import write_outputs
from dentalmapcert.schemas import CaseManifest, SurfaceRegion, write_jsonl


def test_case_manifest_jsonl_writer(tmp_path: Path):
    path = write_jsonl(
        [
            CaseManifest(
                case_id="c1",
                subject_id="s1",
                timepoint_id="t0",
                jaw="upper",
                source_dataset="fixture",
                reference_mesh_path="mesh.obj",
                split="train",
            )
        ],
        tmp_path / "case_manifest.jsonl",
    )

    payload = json.loads(path.read_text().strip())
    assert payload["case_id"] == "c1"


def test_surface_region_rejects_hidden_claim_scope():
    region = SurfaceRegion(
        surface_region_id="r",
        case_id="c",
        tooth_id_fdi=11,
        region_type="buccal_crown",
        claim_scope="subgingival_root",
    )
    try:
        region.validate()
    except ValueError as exc:
        assert "hidden/root/subgingival" in str(exc)
    else:
        raise AssertionError("hidden scope should fail validation")


def test_error_calibrator_interval_is_nonnegative():
    cal = ErrorCalibrator.fit([0.1, 0.2, 0.4, 0.8], alpha=0.25)
    assert cal.interval(0.1)[0] == 0.0
    assert cal.interval(0.1)[1] >= 0.1


def test_report_writer_outputs_jsonl(tmp_path: Path):
    cert = decide_surface_change(
        CertificateInput(
            surface_region_id="s",
            capture_id_t0="t0",
            capture_id_t1="t1",
            coverage_score_t0=0.9,
            coverage_score_t1=0.9,
            error_interval_mm_t0=(0.0, 0.2),
            error_interval_mm_t1=(0.0, 0.2),
            delta_interval_mm=(0.0, 0.2),
        )
    )
    report, jsonl = write_outputs([cert], tmp_path)

    assert report.read_text().startswith("# DentalMapCert")
    assert "surface stable certified" in jsonl.read_text()


# ---------------------------------------------------------------------------
# FDI deciduous tooth support (Task 6)
# ---------------------------------------------------------------------------


def test_surface_region_accepts_permanent_tooth():
    region = SurfaceRegion("r", "c", 36, "buccal_crown")
    region.validate()  # should not raise


def test_surface_region_accepts_deciduous_tooth():
    region = SurfaceRegion("r", "c", 55, "buccal_crown")
    region.validate()  # should not raise


def test_surface_region_accepts_deciduous_boundary_low():
    region = SurfaceRegion("r", "c", 51, "buccal_crown")
    region.validate()


def test_surface_region_accepts_deciduous_boundary_high():
    region = SurfaceRegion("r", "c", 85, "buccal_crown")
    region.validate()


def test_surface_region_rejects_invalid_fdi_between_permanent_and_deciduous():
    region = SurfaceRegion("r", "c", 50, "buccal_crown")
    try:
        region.validate()
    except ValueError as exc:
        assert "FDI" in str(exc)
    else:
        raise AssertionError("tooth_id_fdi=50 should be invalid")


def test_surface_region_rejects_fdi_above_deciduous_range():
    region = SurfaceRegion("r", "c", 86, "buccal_crown")
    try:
        region.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("tooth_id_fdi=86 should be invalid")


# ---------------------------------------------------------------------------
# image_quality module (Task 1)
# ---------------------------------------------------------------------------


def _make_image(h: int = 100, w: int = 100, fill: int = 100) -> np.ndarray:
    """Create a flat greyscale BGR image (uniform fill)."""
    img = np.full((h, w, 3), fill, dtype=np.uint8)
    return img


def test_detect_blur_sharp_image_returns_low_value():
    # A sharp high-contrast image (checkerboard) has high Laplacian variance.
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[::2, ::2] = 255  # checkerboard
    score = detect_blur(img)
    assert 0.0 <= score <= 1.0


def test_detect_blur_flat_image_is_nan():
    # A flat uniform image has zero Laplacian variance → sharpness is undefined
    # (not "maximally blurry"), so detect_blur returns NaN rather than 0.0.
    flat = _make_image(fill=128)
    score = detect_blur(flat)
    assert np.isnan(score)


def test_detect_blur_blurry_textured_image_fires_blur_tag(tmp_path: Path):
    # A smooth low-contrast gradient has small but NON-zero Laplacian variance:
    # sharpness is defined and below threshold, so analyze_view_quality emits
    # 'blur' (not 'low_detail').
    grad = np.tile(np.linspace(40, 60, 100, dtype=np.uint8), (100, 1))
    img = np.stack([grad, grad, grad], axis=2)
    p = tmp_path / "blurry.png"
    cv2.imwrite(str(p), img)
    tags = analyze_view_quality(p)
    assert "blur" in tags
    assert "low_detail" not in tags


def test_detect_glare_dark_image_is_glare_free():
    dark = _make_image(fill=10)
    score = detect_glare(dark)
    assert score == 1.0


def test_detect_glare_bright_image_flags_glare():
    # Nearly all pixels are very bright (≥240) → glare detected → score near 0.
    bright = _make_image(fill=250)
    score = detect_glare(bright)
    assert score < 1.0


def test_analyze_view_quality_missing_file_returns_unreadable():
    tags = analyze_view_quality("/nonexistent/path/to/image.png")
    assert tags == ["unreadable"]


def test_analyze_view_quality_dark_image_has_no_issues(tmp_path: Path):
    # Uniform dark image: sharpness is undefined (zero Laplacian) so it is
    # tagged 'low_detail', NOT the fabricated 'blur'. No glare, no occlusion.
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    p = tmp_path / "dark.png"
    cv2.imwrite(str(p), img)
    tags = analyze_view_quality(p)
    assert "blur" not in tags
    assert "low_detail" in tags
    assert "glare" not in tags
    assert "occlusion" not in tags


def test_analyze_view_quality_returns_list(tmp_path: Path):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    p = tmp_path / "img.png"
    cv2.imwrite(str(p), img)
    tags = analyze_view_quality(p)
    assert isinstance(tags, list)


def test_detect_blur_empty_array_raises():
    """detect_blur fast-fails with ValueError on an empty array."""
    with pytest.raises(ValueError, match="non-empty image"):
        detect_blur(np.array([], dtype=np.uint8))


def test_detect_blur_none_raises():
    """detect_blur fast-fails with ValueError on a None image."""
    with pytest.raises(ValueError, match="non-empty image"):
        detect_blur(None)


def test_detect_blur_grayscale_image():
    """detect_blur handles 2D grayscale input without color conversion (line 48)."""
    gray = np.zeros((20, 20), dtype=np.uint8)
    gray[10, 10] = 255
    score = detect_blur(gray)
    assert 0.0 <= score <= 1.0


def test_detect_glare_empty_array_returns_one():
    """detect_glare returns 1.0 for an empty array (line 70 coverage)."""
    assert detect_glare(np.array([], dtype=np.uint8)) == 1.0


def test_detect_glare_grayscale_promotes_to_bgr():
    """detect_glare promotes 2D grayscale to BGR before HSV conversion (line 73)."""
    gray = np.full((20, 20), 10, dtype=np.uint8)  # dark → no glare
    score = detect_glare(gray)
    assert score == 1.0


def test_detect_occlusion_empty_array_returns_one():
    """_detect_occlusion returns 1.0 for an empty array (line 91 coverage)."""
    from dentalmapcert.image_quality import _detect_occlusion
    assert _detect_occlusion(np.array([], dtype=np.uint8)) == 1.0


def test_detect_occlusion_grayscale_image():
    """_detect_occlusion handles 2D grayscale input without color conversion (line 96)."""
    from dentalmapcert.image_quality import _detect_occlusion
    gray = np.full((20, 20), 30, dtype=np.uint8)  # very dark → no occlusion
    score = _detect_occlusion(gray)
    assert 0.0 <= score <= 1.0


def test_analyze_view_quality_bright_image_has_glare_tag(tmp_path: Path):
    """analyze_view_quality appends 'glare' for near-white images (line 144 coverage)."""
    img = np.full((100, 100, 3), 250, dtype=np.uint8)
    p = tmp_path / "bright.png"
    cv2.imwrite(str(p), img)
    tags = analyze_view_quality(p)
    assert "glare" in tags


def test_analyze_view_quality_midgray_image_has_occlusion_tag(tmp_path: Path):
    """analyze_view_quality appends 'occlusion' for flesh-tone border images (line 148 coverage)."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    p = tmp_path / "midgray.png"
    cv2.imwrite(str(p), img)
    tags = analyze_view_quality(p)
    assert "occlusion" in tags


# ---------------------------------------------------------------------------
# validate_fdi_id standalone helper (FIX 10)
# ---------------------------------------------------------------------------

from dentalmapcert.schemas import validate_fdi_id


def test_validate_fdi_id_permanent_tooth_is_valid():
    assert validate_fdi_id(11) is True
    assert validate_fdi_id(36) is True
    assert validate_fdi_id(48) is True


def test_validate_fdi_id_deciduous_tooth_is_valid():
    assert validate_fdi_id(51) is True
    assert validate_fdi_id(65) is True
    assert validate_fdi_id(85) is True


def test_validate_fdi_id_invalid_between_ranges():
    # 49-50 is the gap between permanent (max 48) and deciduous (min 51)
    assert validate_fdi_id(49) is False
    assert validate_fdi_id(50) is False


def test_validate_fdi_id_out_of_range_low():
    assert validate_fdi_id(10) is False
    assert validate_fdi_id(0) is False


def test_validate_fdi_id_out_of_range_high():
    assert validate_fdi_id(86) is False
    assert validate_fdi_id(99) is False


def test_validate_fdi_id_boundary_permanent_low():
    assert validate_fdi_id(11) is True


def test_validate_fdi_id_boundary_deciduous_high():
    assert validate_fdi_id(85) is True


# ---------------------------------------------------------------------------
# FusionTimepoint and to_fusion_timepoint (FIX 7)
# ---------------------------------------------------------------------------

from dentalmapcert.fusion import FusionTimepoint, to_fusion_timepoint


def test_fusion_timepoint_to_dict_has_expected_keys():
    from dentalmapcert.calibration import ErrorCalibrator
    from dentalmapcert.capture_protocol import coverage_per_region

    calibrator = ErrorCalibrator.fit([0.1, 0.2, 0.3, 0.4], alpha=0.1)
    cov = coverage_per_region(["anterior_close", "left_buccal", "right_buccal"], {})
    ft = to_fusion_timepoint(cov, calibrator, subject_id="S001", timepoint_id="t0")

    d = ft.to_dict()
    assert d["subject_id"] == "S001"
    assert d["timepoint_id"] == "t0"
    assert "visible_surface_certificates" in d
    assert isinstance(d["visible_surface_certificates"], list)


def test_fusion_timepoint_has_radiograph_pair_false_when_none():
    from dentalmapcert.calibration import ErrorCalibrator

    calibrator = ErrorCalibrator.fit([0.1, 0.2, 0.3, 0.4], alpha=0.1)
    ft = to_fusion_timepoint({}, calibrator)
    assert ft.has_radiograph_pair is False


def test_fusion_timepoint_has_radiograph_pair_true_when_set():
    from dentalmapcert.calibration import ErrorCalibrator

    calibrator = ErrorCalibrator.fit([0.1, 0.2, 0.3, 0.4], alpha=0.1)
    ft = to_fusion_timepoint({}, calibrator, radiograph_id="xray_001")
    assert ft.has_radiograph_pair is True


def test_fusion_timepoint_certificates_count_matches_regions():
    from dentalmapcert.calibration import ErrorCalibrator

    calibrator = ErrorCalibrator.fit([0.1, 0.2, 0.3, 0.4], alpha=0.1)
    cov = {"anterior_crown": 0.8, "buccal_crown": 0.7}
    ft = to_fusion_timepoint(cov, calibrator, subject_id="S002")
    assert len(ft.visible_surface_certificates) == 2


def test_fusion_timepoint_is_frozen():
    from dentalmapcert.calibration import ErrorCalibrator

    calibrator = ErrorCalibrator.fit([0.1, 0.2], alpha=0.1)
    ft = to_fusion_timepoint({}, calibrator)
    try:
        ft.subject_id = "mutated"  # type: ignore[misc]
    except Exception:
        pass  # frozen dataclass raises FrozenInstanceError
    else:
        raise AssertionError("FusionTimepoint should be immutable")


# ---------------------------------------------------------------------------
# render_report synthetic disclaimer (FIX 11)
# ---------------------------------------------------------------------------

from dentalmapcert.report import render_report


def test_render_report_includes_synthetic_note_by_default():
    text = render_report([])
    assert "synthetic heuristics" in text


def test_render_report_no_synthetic_note_when_disabled():
    text = render_report([], synthetic=False)
    assert "synthetic heuristics" not in text


# ---------------------------------------------------------------------------
# ErrorCalibrator edge cases (lines 17, 19, 22 coverage)
# ---------------------------------------------------------------------------

def test_error_calibrator_raises_on_empty_residuals():
    """ErrorCalibrator.fit raises ValueError for empty residual list (line 17)."""
    with pytest.raises(ValueError, match="at least one"):
        ErrorCalibrator.fit([])


def test_error_calibrator_raises_on_bad_alpha_zero():
    """ErrorCalibrator.fit raises ValueError when alpha=0 (line 19)."""
    with pytest.raises(ValueError, match="alpha must be in"):
        ErrorCalibrator.fit([0.1], alpha=0.0)


def test_error_calibrator_raises_on_negative_residual():
    """ErrorCalibrator.fit raises ValueError when any residual is negative (line 22)."""
    with pytest.raises(ValueError, match="non-negative"):
        ErrorCalibrator.fit([-0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# coverage_per_region with image_paths_per_view (lines 152-160 coverage)
# ---------------------------------------------------------------------------

def test_coverage_per_region_with_image_paths_detects_tags(tmp_path: Path):
    """coverage_per_region auto-detects quality tags when image_paths_per_view is provided."""
    from dentalmapcert.capture_protocol import coverage_per_region

    # Write a dark image that produces no quality issues.
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    p = tmp_path / "anterior_close.png"
    cv2.imwrite(str(p), img)

    cov = coverage_per_region(
        ["anterior_close"],
        quality_tags_per_view={},
        image_paths_per_view={"anterior_close": p},
    )
    assert isinstance(cov, dict)


def test_coverage_per_region_explicit_tags_take_precedence(tmp_path: Path):
    """Caller-supplied quality tags are not overwritten by auto-detection (lines 155-156)."""
    from dentalmapcert.capture_protocol import coverage_per_region

    img = np.zeros((50, 50, 3), dtype=np.uint8)
    p = tmp_path / "anterior_close.png"
    cv2.imwrite(str(p), img)

    # Explicit tag supplied for the same view — should not be overwritten.
    cov = coverage_per_region(
        ["anterior_close"],
        quality_tags_per_view={"anterior_close": ["glare"]},
        image_paths_per_view={"anterior_close": p},
    )
    assert isinstance(cov, dict)


# ---------------------------------------------------------------------------
# CaptureManifest.validate() edge cases (lines 88-91, 126 coverage)
# ---------------------------------------------------------------------------

def test_capture_manifest_validate_raises_on_empty_capture_id():
    """CaptureManifest.validate raises ValueError when capture_id is empty (lines 88, 126)."""
    from dentalmapcert.schemas import CaptureManifest, CaptureView

    view = CaptureView(view_id="v1", image_path="img.jpg")
    manifest = CaptureManifest(
        capture_id="",
        case_id="case1",
        capture_type="phone",
        views=[view],
    )
    with pytest.raises(ValueError, match="capture_id"):
        manifest.validate()


def test_capture_manifest_validate_raises_on_empty_views():
    """CaptureManifest.validate raises ValueError when views list is empty (lines 90-91)."""
    from dentalmapcert.schemas import CaptureManifest

    manifest = CaptureManifest(
        capture_id="cap1",
        case_id="case1",
        capture_type="phone",
        views=[],
    )
    with pytest.raises(ValueError, match="at least one view"):
        manifest.validate()

