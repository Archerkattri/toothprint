import pytest
from dentalmapcert.certificate import CertificateInput, CertificateOutput, decide_surface_change, recapture_actions


def test_stable_certificate_when_delta_interval_is_small_and_visible():
    result = decide_surface_change(
        CertificateInput(
            surface_region_id="s",
            capture_id_t0="t0",
            capture_id_t1="t1",
            coverage_score_t0=0.9,
            coverage_score_t1=0.9,
            error_interval_mm_t0=(0.0, 0.2),
            error_interval_mm_t1=(0.0, 0.2),
            delta_interval_mm=(0.0, 0.3),
            region_type="buccal_crown",
        )
    )

    assert result.label == "surface stable certified"


def test_change_certificate_when_delta_lower_bound_exceeds_threshold():
    result = decide_surface_change(
        CertificateInput(
            surface_region_id="s",
            capture_id_t0="t0",
            capture_id_t1="t1",
            coverage_score_t0=0.91,
            coverage_score_t1=0.86,
            error_interval_mm_t0=(0.0, 0.2),
            error_interval_mm_t1=(0.0, 0.2),
            delta_interval_mm=(0.9, 1.4),
            region_type="anterior_crown",
        )
    )

    assert result.label == "surface change certified"


def test_low_coverage_routes_to_recapture_with_actions():
    result = decide_surface_change(
        CertificateInput(
            surface_region_id="s",
            capture_id_t0="t0",
            capture_id_t1="t1",
            coverage_score_t0=0.4,
            coverage_score_t1=0.7,
            error_interval_mm_t0=(0.0, 0.5),
            error_interval_mm_t1=(0.0, 0.5),
            delta_interval_mm=(0.0, 0.8),
            quality_tags_t0=("glare", "left_buccal_missing"),
        )
    )

    assert result.label == "uncertain / recapture"
    assert "reduce_glare" in result.recapture_actions
    assert "need_left_buccal_view" in result.recapture_actions


def test_hidden_region_is_not_claimable():
    result = decide_surface_change(
        CertificateInput(
            surface_region_id="s",
            capture_id_t0="t0",
            capture_id_t1="t1",
            coverage_score_t0=0.99,
            coverage_score_t1=0.99,
            error_interval_mm_t0=(0.0, 0.1),
            error_interval_mm_t1=(0.0, 0.1),
            delta_interval_mm=(1.2, 1.8),
            region_type="subgingival_root",
        )
    )

    assert result.label == "not visible / not claimable"


def _make_input(**kwargs) -> CertificateInput:
    defaults = dict(
        surface_region_id="s",
        capture_id_t0="t0",
        capture_id_t1="t1",
        coverage_score_t0=0.9,
        coverage_score_t1=0.9,
        error_interval_mm_t0=(0.0, 0.1),
        error_interval_mm_t1=(0.0, 0.1),
        delta_interval_mm=(0.0, 0.3),
    )
    defaults.update(kwargs)
    return CertificateInput(**defaults)


def test_decide_raises_when_stable_threshold_ge_change_threshold():
    """ValueError raised when stable_threshold_mm >= change_threshold_mm (line 86)."""
    item = _make_input()
    with pytest.raises(ValueError, match="stable_threshold_mm"):
        decide_surface_change(item, stable_threshold_mm=1.0, change_threshold_mm=0.5)


def test_not_claimable_item_gets_not_claimable_label():
    """item.claimable=False → label 'not visible / not claimable' (lines 91-92)."""
    item = _make_input(claimable=False)
    result = decide_surface_change(item)
    assert result.label == "not visible / not claimable"
    assert result.not_claimable_reason == "surface region is marked non-claimable"


def test_recapture_actions_right_buccal_missing_tag():
    """'right_buccal_missing' quality tag → 'need_right_buccal_view' action (line 132)."""
    item = _make_input(
        coverage_score_t0=0.3,
        coverage_score_t1=0.3,
        quality_tags_t0=("right_buccal_missing",),
    )
    actions = recapture_actions(item, coverage_threshold=0.7)
    assert "need_right_buccal_view" in actions


def test_recapture_actions_anterior_missing_tag():
    """'anterior_missing' quality tag → 'need_anterior_close_view' action (line 134)."""
    item = _make_input(
        coverage_score_t0=0.3,
        coverage_score_t1=0.3,
        quality_tags_t0=("anterior_missing",),
    )
    actions = recapture_actions(item, coverage_threshold=0.7)
    assert "need_anterior_close_view" in actions


def test_recapture_actions_upper_occlusal_missing_tag():
    """'upper_occlusal_missing' tag → 'need_upper_occlusal_view' action (line 136)."""
    item = _make_input(
        coverage_score_t0=0.3,
        coverage_score_t1=0.3,
        quality_tags_t0=("upper_occlusal_missing",),
    )
    actions = recapture_actions(item, coverage_threshold=0.7)
    assert "need_upper_occlusal_view" in actions


def test_recapture_actions_lower_occlusal_missing_tag():
    """'lower_occlusal_missing' tag → 'need_lower_occlusal_view' action (line 138)."""
    item = _make_input(
        coverage_score_t0=0.3,
        coverage_score_t1=0.3,
        quality_tags_t0=("lower_occlusal_missing",),
    )
    actions = recapture_actions(item, coverage_threshold=0.7)
    assert "need_lower_occlusal_view" in actions


def test_recapture_actions_blur_tag_adds_focus_action():
    """'blur' tag triggers 'increase_focus_or_distance' even when coverage is fine (line 142)."""
    item = _make_input(
        coverage_score_t0=0.3,
        coverage_score_t1=0.3,
        quality_tags_t0=("blur",),
    )
    actions = recapture_actions(item, coverage_threshold=0.7)
    assert "increase_focus_or_distance" in actions


def test_recapture_actions_lip_occlusion_tag():
    """'lip_occlusion' tag → 'move_cheek_or_lip' action (line 144)."""
    item = _make_input(quality_tags_t0=("lip_occlusion",))
    actions = recapture_actions(item, coverage_threshold=0.7)
    assert "move_cheek_or_lip" in actions


def test_certificate_input_raises_on_coverage_out_of_range():
    """CertificateInput raises ValueError when coverage_score > 1.0 (line 155 via __post_init__)."""
    with pytest.raises(ValueError, match="coverage_score_t0"):
        _make_input(coverage_score_t0=1.5)


def test_certificate_input_raises_on_interval_wrong_length():
    """CertificateInput raises ValueError when error_interval has wrong length (line 160)."""
    with pytest.raises(ValueError, match="error_interval_mm_t0"):
        _make_input(error_interval_mm_t0=(0.0, 0.1, 0.2))  # type: ignore[arg-type]


def test_certificate_input_raises_on_interval_hi_less_than_lo():
    """CertificateInput raises ValueError when interval hi < lo (line 163)."""
    with pytest.raises(ValueError, match="delta_interval_mm"):
        _make_input(delta_interval_mm=(0.5, 0.1))
