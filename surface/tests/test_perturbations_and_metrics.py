"""Tests for perturbations, eval_metrics, and baselines modules."""
from __future__ import annotations

import numpy as np
import pytest

from dentalmapcert.certificate import CertificateInput, CertificateOutput, decide_surface_change
from dentalmapcert.perturbations import (
    apply_all,
    missing_view,
    partial_occlusion,
    pose_jitter,
    sparse_dropout,
    surface_noise,
)
from dentalmapcert.eval_metrics import BenchmarkMetrics, compute_metrics, coverage_vs_false_change_curve, _roc_auc
from dentalmapcert.baselines import coverage_only_baseline, naive_baseline, uncertainty_only_baseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_points(n: int = 100, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)).astype(float)


def _make_input(
    *,
    coverage_t0: float = 0.9,
    coverage_t1: float = 0.9,
    delta: tuple[float, float] = (0.0, 0.2),
    surface_id: str = "s1",
    t0: str = "c0",
    t1: str = "c1",
) -> CertificateInput:
    return CertificateInput(
        surface_region_id=surface_id,
        capture_id_t0=t0,
        capture_id_t1=t1,
        coverage_score_t0=coverage_t0,
        coverage_score_t1=coverage_t1,
        error_interval_mm_t0=(0.0, 0.1),
        error_interval_mm_t1=(0.0, 0.1),
        delta_interval_mm=delta,
    )


def _make_output(
    *,
    label: str = "surface stable certified",
    coverage_t0: float = 0.9,
    coverage_t1: float = 0.9,
    delta: tuple[float, float] = (0.0, 0.2),
    surface_id: str = "s1",
    recapture_actions: list | None = None,
) -> CertificateOutput:
    return CertificateOutput(
        certificate_id=f"cert_{surface_id}_c0_c1",
        surface_region_id=surface_id,
        capture_id_t0="c0",
        capture_id_t1="c1",
        coverage_score_t0=coverage_t0,
        coverage_score_t1=coverage_t1,
        error_interval_mm_t0=(0.0, 0.1),
        error_interval_mm_t1=(0.0, 0.1),
        delta_interval_mm=delta,
        label=label,
        recapture_actions=recapture_actions or [],
    )


# ===========================================================================
# Perturbation tests (8)
# ===========================================================================

def test_pose_jitter_preserves_point_count():
    pts = _make_points(200)
    result = pose_jitter(pts)
    assert result.points.shape == pts.shape


def test_pose_jitter_zero_params_returns_same_points():
    pts = _make_points(50)
    result = pose_jitter(pts, rotation_deg=0.0, translation_mm=0.0, seed=0)
    np.testing.assert_allclose(result.points, pts, atol=1e-10)


def test_sparse_dropout_half_points():
    pts = _make_points(200)
    result = sparse_dropout(pts, dropout_fraction=0.5, seed=7)
    assert result.points.shape[0] == 100


def test_surface_noise_zero_std_returns_identical():
    pts = _make_points(80)
    result = surface_noise(pts, noise_std_mm=0.0, seed=0)
    np.testing.assert_array_equal(result.points, pts)


def test_partial_occlusion_drops_exactly_fraction():
    pts = _make_points(100)
    fraction = 0.3
    result = partial_occlusion(pts, axis=0, fraction=fraction)
    expected_kept = 100 - int(100 * fraction)
    assert result.points.shape[0] == expected_kept


def test_apply_all_returns_four_results():
    pts = _make_points(50)
    results = apply_all(pts, seed=1)
    assert len(results) == 5


def test_perturbation_family_strings():
    pts = _make_points(50)
    assert pose_jitter(pts).family == "pose_jitter"
    assert sparse_dropout(pts).family == "sparse_dropout"
    assert surface_noise(pts).family == "surface_noise"
    assert partial_occlusion(pts).family == "partial_occlusion"
    assert missing_view(pts).family == "missing_view"


def test_perturbations_reproducible_with_same_seed():
    pts = _make_points(100)
    r1 = pose_jitter(pts, seed=42)
    r2 = pose_jitter(pts, seed=42)
    np.testing.assert_array_equal(r1.points, r2.points)

    r3 = sparse_dropout(pts, seed=99)
    r4 = sparse_dropout(pts, seed=99)
    np.testing.assert_array_equal(r3.points, r4.points)


# ===========================================================================
# Eval metrics tests (6)
# ===========================================================================

def test_compute_metrics_empty_list():
    m = compute_metrics([])
    assert m.n == 0
    assert m.capture_only_false_change_rate == 0.0
    assert m.useful_certified_coverage == 0.0
    assert m.uncertain_rate == 0.0
    assert m.mean_delta_width_mm == 0.0
    assert m.delta_width_std_mm == 0.0
    assert m.uncertainty_auc == 0.0
    assert m.recapture_trigger_rate == 0.0


def test_compute_metrics_all_stable_certified():
    outputs = [_make_output(label="surface stable certified") for _ in range(5)]
    m = compute_metrics(outputs, true_labels=["stable"] * 5)
    assert m.useful_certified_coverage == 1.0
    assert m.capture_only_false_change_rate == 0.0


def test_compute_metrics_false_change_rate_one():
    outputs = [_make_output(label="surface change certified")]
    m = compute_metrics(outputs, true_labels=["stable"])
    assert m.capture_only_false_change_rate == 1.0


def test_coverage_vs_false_change_curve_returns_21_dicts():
    outputs = [_make_output(coverage_t0=0.8, coverage_t1=0.8)]
    curve = coverage_vs_false_change_curve(outputs)
    assert len(curve) == 21


def test_coverage_vs_false_change_curve_keys():
    outputs = [_make_output()]
    curve = coverage_vs_false_change_curve(outputs)
    for entry in curve:
        assert "coverage_threshold" in entry
        assert "useful_coverage" in entry
        assert "false_change_rate" in entry


def test_coverage_vs_false_change_curve_threshold_boundaries():
    # At threshold=0.0 the item with coverage 0.8 is certified; at threshold=1.0 it is not
    outputs = [_make_output(coverage_t0=0.8, coverage_t1=0.8, label="surface stable certified")]
    curve = coverage_vs_false_change_curve(outputs)

    # threshold=0.0 (first entry): item is "certified", useful_coverage should be 1.0
    assert curve[0]["coverage_threshold"] == pytest.approx(0.0)
    assert curve[0]["useful_coverage"] == pytest.approx(1.0)

    # threshold=1.0 (last entry): item with coverage 0.8 does NOT meet threshold
    assert curve[-1]["coverage_threshold"] == pytest.approx(1.0)
    assert curve[-1]["useful_coverage"] == pytest.approx(0.0)


# ===========================================================================
# Baselines tests (6)
# ===========================================================================

def test_naive_baseline_always_stable():
    for cov in [0.0, 0.5, 0.99]:
        item = _make_input(coverage_t0=cov, coverage_t1=cov, delta=(0.0, 5.0))
        out = naive_baseline(item)
        assert out.label == "surface stable certified"


def test_coverage_only_baseline_high_coverage_stable():
    item = _make_input(coverage_t0=0.9, coverage_t1=0.9)
    out = coverage_only_baseline(item, coverage_threshold=0.75)
    assert out.label == "surface stable certified"


def test_coverage_only_baseline_low_coverage_recapture():
    item = _make_input(coverage_t0=0.5, coverage_t1=0.5)
    out = coverage_only_baseline(item, coverage_threshold=0.75)
    assert out.label == "uncertain / recapture"


def test_uncertainty_only_baseline_narrow_delta_stable():
    item = _make_input(delta=(0.0, 0.2))
    out = uncertainty_only_baseline(item, stable_threshold_mm=0.35)
    assert out.label == "surface stable certified"


def test_uncertainty_only_baseline_wide_delta_recapture():
    item = _make_input(delta=(0.0, 1.0))
    out = uncertainty_only_baseline(item, stable_threshold_mm=0.35)
    assert out.label == "uncertain / recapture"


def test_all_baselines_return_valid_certificate_output():
    item = _make_input(surface_id="tooth_11", t0="visit_A", t1="visit_B")
    for fn in (naive_baseline, coverage_only_baseline, uncertainty_only_baseline):
        out = fn(item)
        assert isinstance(out, CertificateOutput)
        assert out.surface_region_id == "tooth_11"
        assert out.certificate_id == "cert_tooth_11_visit_A_visit_B"
        assert out.label in (
            "surface stable certified",
            "surface change certified",
            "uncertain / recapture",
            "not visible / not claimable",
        )


# ===========================================================================
# missing_view tests (new perturbation family)
# ===========================================================================

def test_missing_view_drops_top_z_fraction():
    """missing_view with view_direction=(0,0,1) should drop the top 30% by z."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 3))
    result = missing_view(pts, view_direction=(0.0, 0.0, 1.0), occlusion_fraction=0.30)
    # 30% dropped → ~70% kept; due to percentile tie-breaking allow small tolerance
    expected_kept = int(200 * 0.70)
    assert abs(result.points.shape[0] - expected_kept) <= 2
    # Remaining points should all have z below the 70th percentile of original
    z_threshold = np.percentile(pts[:, 2], 70.0)
    assert np.all(result.points[:, 2] < z_threshold + 1e-10)


def test_missing_view_family_string():
    pts = _make_points(50)
    result = missing_view(pts)
    assert result.family == "missing_view"


def test_missing_view_empty_points():
    pts = np.zeros((0, 3))
    result = missing_view(pts)
    assert result.points.shape == (0, 3)
    assert result.family == "missing_view"


def test_apply_all_returns_five_results():
    pts = _make_points(50)
    results = apply_all(pts, seed=0)
    assert len(results) == 5
    families = [r.family for r in results]
    assert "missing_view" in families


# ===========================================================================
# _roc_auc tests
# ===========================================================================

def test_roc_auc_perfect_discrimination():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array([0, 0, 1, 1])
    assert _roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_random():
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert _roc_auc(scores, labels) == pytest.approx(0.5)


def test_roc_auc_no_positives():
    scores = np.array([1.0, 2.0])
    labels = np.array([0, 0])
    assert _roc_auc(scores, labels) == 0.0


def test_roc_auc_no_negatives():
    scores = np.array([1.0, 2.0])
    labels = np.array([1, 1])
    assert _roc_auc(scores, labels) == 0.0


# ===========================================================================
# compute_metrics uncertainty_auc tests
# ===========================================================================

def test_compute_metrics_uncertainty_auc_with_discrimination():
    """Narrow-width outputs labeled stable, wide-width outputs labeled changed
    should yield AUC > 0.5."""
    # Narrow delta → high confidence → should score well for "stable"
    narrow = [_make_output(delta=(0.0, 0.1), label="surface stable certified") for _ in range(4)]
    # Wide delta → low confidence → should score poorly for "stable"
    wide = [_make_output(delta=(0.0, 2.0), label="surface stable certified") for _ in range(4)]
    outputs = narrow + wide
    true_labels = ["stable"] * 4 + ["changed"] * 4
    m = compute_metrics(outputs, true_labels=true_labels)
    assert m.uncertainty_auc > 0.5


def test_compute_metrics_uncertainty_auc_none_labels():
    """When true_labels is None, uncertainty_auc should be 0.0."""
    outputs = [_make_output(delta=(0.0, 0.5)) for _ in range(3)]
    m = compute_metrics(outputs, true_labels=None)
    assert m.uncertainty_auc == 0.0


def test_compute_metrics_no_stable_labels_false_change_rate_zero():
    """When all true_labels are 'changed' (n_stable==0), false_change_rate is 0.0 (line 78)."""
    outputs = [_make_output(label="surface change certified") for _ in range(3)]
    m = compute_metrics(outputs, true_labels=["changed"] * 3)
    assert m.capture_only_false_change_rate == 0.0


# ---------------------------------------------------------------------------
# Audit test-quality additions
# ---------------------------------------------------------------------------

def test_missing_view_diagonal_drops_top_along_arbitrary_direction():
    """missing_view must drop the contiguous top of the projection onto the
    GIVEN direction, proving it projects onto view_direction (not a hardcoded
    axis)."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 3))
    vd = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    res = missing_view(pts, view_direction=(1.0, 1.0, 1.0), occlusion_fraction=0.25)

    assert abs(res.points.shape[0] - 150) <= 2  # ~75% kept

    # The kept points must be exactly the lowest-projection 75% along vd: the
    # max projection among kept points is below the min among dropped points.
    proj_all = pts @ vd
    kept_proj = res.points @ vd
    threshold = np.sort(proj_all)[res.points.shape[0] - 1]
    assert kept_proj.max() <= threshold + 1e-9


@pytest.mark.parametrize("fn", [pose_jitter, sparse_dropout, surface_noise, partial_occlusion, missing_view])
def test_perturbation_empty_cloud_is_clean(fn):
    """Every perturbation handles an empty cloud without raising/warning and
    returns an empty (0,3) result."""
    import warnings

    pts = np.zeros((0, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = fn(pts)
    assert res.points.shape == (0, 3)


def test_roc_auc_partial_ties():
    # pos={2,3}, neg={1,2}: (2>1)=1, (2==2)=0.5, (3>1)=1, (3>2)=1 -> 3.5/4.
    scores = np.array([1.0, 2.0, 2.0, 3.0])
    labels = np.array([0, 0, 1, 1])
    assert _roc_auc(scores, labels) == pytest.approx(3.5 / 4)


def test_compute_metrics_delta_width_std_is_population():
    # Two certs with delta widths 0.2 and 0.6: population std (ddof=0) = 0.2,
    # sample std (ddof=1) would be ~0.2828. Pin the population convention.
    outs = [
        _make_output(delta=(0.0, 0.2), surface_id="a"),
        _make_output(delta=(0.0, 0.6), surface_id="b"),
    ]
    m = compute_metrics(outs)
    assert m.delta_width_std_mm == pytest.approx(0.2, abs=1e-9)
    assert m.delta_width_std_mm != pytest.approx(0.2828, abs=1e-3)
