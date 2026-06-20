"""Tests for coverage_from_point_cloud and longitudinal pairing."""

from __future__ import annotations

import pytest

from dentalmapcert.coverage import CoverageScore, coverage_from_point_cloud, _adaptive_grid
from dentalmapcert.longitudinal import LongitudinalPair, pair_by_subject


# ---------------------------------------------------------------------------
# coverage_from_point_cloud
# ---------------------------------------------------------------------------

_BBOX = (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)  # 10mm cube


def _dense_grid(n_per_axis: int = 10) -> list[tuple[float, float, float]]:
    """Return n_per_axis^3 evenly-spaced points filling the bbox."""
    step = 10.0 / n_per_axis
    points = []
    for i in range(n_per_axis):
        for j in range(n_per_axis):
            for k in range(n_per_axis):
                points.append((i * step + step / 2, j * step + step / 2, k * step + step / 2))
    return points


class TestCoverageFromPointCloud:
    def test_returns_coverage_score_instance(self):
        score = coverage_from_point_cloud("r1", _dense_grid(), _BBOX)
        assert isinstance(score, CoverageScore)

    def test_method_is_reconstruction(self):
        score = coverage_from_point_cloud("r1", _dense_grid(), _BBOX)
        assert score.method == "reconstruction"

    def test_surface_region_id_is_preserved(self):
        score = coverage_from_point_cloud("my_region", _dense_grid(), _BBOX)
        assert score.surface_region_id == "my_region"

    def test_dense_uniform_grid_approaches_one(self):
        # 10x10x10 points centred in each voxel → should fill all 1000 voxels → 0.95 cap.
        score = coverage_from_point_cloud("r1", _dense_grid(10), _BBOX)
        assert score.coverage_fraction == pytest.approx(0.95)

    def test_dense_grid_occupies_all_voxels(self):
        score = coverage_from_point_cloud("r1", _dense_grid(10), _BBOX)
        assert score.stable_pixels == 1000
        assert score.total_pixels == 1000

    def test_sparse_point_cloud_gives_lower_coverage(self):
        # Only 8 corner points — at most 8 voxels occupied.
        # With adaptive grid (< 100 pts → GRID=5, 125 total voxels):
        # 8 / 125 = 0.064 < 0.10.
        corners = [
            (0.5, 0.5, 0.5), (9.5, 0.5, 0.5),
            (0.5, 9.5, 0.5), (9.5, 9.5, 0.5),
            (0.5, 0.5, 9.5), (9.5, 0.5, 9.5),
            (0.5, 9.5, 9.5), (9.5, 9.5, 9.5),
        ]
        score = coverage_from_point_cloud("r1", corners, _BBOX)
        assert score.coverage_fraction < 0.10  # 8/125 = 0.064 with GRID=5

    def test_empty_point_cloud_gives_zero_coverage(self):
        score = coverage_from_point_cloud("r1", [], _BBOX)
        assert score.coverage_fraction == 0.0
        assert score.stable_pixels == 0

    def test_empty_point_cloud_total_pixels_is_adaptive(self):
        # Empty cloud → 0 points < 100 → GRID=5 → 125 voxels.
        score = coverage_from_point_cloud("r1", [], _BBOX)
        assert score.total_pixels == 125

    def test_explicit_grid_overrides_adaptive(self):
        # A large cloud would pick GRID=10 (1000 voxels) adaptively, but an
        # explicit grid=5 pins it to 125 voxels for cross-record comparability.
        score = coverage_from_point_cloud("r1", _dense_grid(10), _BBOX, grid=5)
        assert score.total_pixels == 125

    def test_explicit_grid_makes_clouds_comparable(self):
        # Same fixed grid → same total_pixels regardless of point count.
        small = coverage_from_point_cloud("r1", _dense_grid(4), _BBOX, grid=8)
        large = coverage_from_point_cloud("r2", _dense_grid(12), _BBOX, grid=8)
        assert small.total_pixels == large.total_pixels == 512

    def test_invalid_grid_raises(self):
        with pytest.raises(ValueError, match="grid must be a positive integer"):
            coverage_from_point_cloud("r1", _dense_grid(), _BBOX, grid=0)

    def test_point_on_bbox_max_face_is_counted_not_indexerror(self):
        # A point exactly at x_max=10 passes the inclusive bbox filter (<= x_max)
        # and must be clamped into the final voxel index (min(GRID-1, ...)),
        # not index out of range. Guards against a `<=`->`<` filter regression
        # or removal of the index clamp.
        score = coverage_from_point_cloud("r1", [(10.0, 5.0, 5.0)], _BBOX)
        assert score.stable_pixels == 1
        assert score.coverage_fraction > 0.0

    def test_points_outside_bbox_are_ignored(self):
        outside_points = [(-5.0, -5.0, -5.0), (100.0, 100.0, 100.0)]
        score = coverage_from_point_cloud("r1", outside_points, _BBOX)
        assert score.coverage_fraction == 0.0

    def test_coverage_capped_at_0_95(self):
        # Even with vastly more points than voxels the cap holds.
        many_points = [(float(i % 10) + 0.5, float((i // 10) % 10) + 0.5, float((i // 100) % 10) + 0.5)
                       for i in range(5000)]
        score = coverage_from_point_cloud("r1", many_points, _BBOX)
        assert score.coverage_fraction <= 0.95

    def test_coverage_fraction_in_unit_interval(self):
        score = coverage_from_point_cloud("r1", _dense_grid(5), _BBOX)
        assert 0.0 <= score.coverage_fraction <= 1.0

    def test_degenerate_bbox_returns_zero(self):
        # Zero-volume bbox (x_min == x_max).
        degenerate_bbox = (5.0, 0.0, 0.0, 5.0, 10.0, 10.0)
        score = coverage_from_point_cloud("r1", _dense_grid(), degenerate_bbox)
        assert score.coverage_fraction == 0.0

    def test_invalid_bbox_raises_value_error(self):
        with pytest.raises(ValueError, match="region_bbox must have exactly 6"):
            coverage_from_point_cloud("r1", [], (0.0, 0.0, 0.0))

    def test_single_point_inside_bbox_gives_nonzero_coverage(self):
        score = coverage_from_point_cloud("r1", [(5.0, 5.0, 5.0)], _BBOX)
        assert score.coverage_fraction > 0.0
        assert score.stable_pixels == 1

    def test_partial_coverage_is_proportional(self):
        # Fill only one 10th of the z-axis — expect roughly 10% coverage (100/1000 voxels).
        points = [
            (float(i) + 0.5, float(j) + 0.5, 0.5)
            for i in range(10) for j in range(10)
        ]
        score = coverage_from_point_cloud("r1", points, _BBOX)
        # 100 unique voxels in z-layer 0 out of 1000 → 0.1 (100 pts → GRID=10)
        assert score.stable_pixels == 100
        assert score.coverage_fraction == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# _adaptive_grid — resolution selection logic
# ---------------------------------------------------------------------------

class TestAdaptiveGrid:
    def test_small_cloud_gives_grid_5(self):
        assert _adaptive_grid(0) == 5
        assert _adaptive_grid(99) == 5

    def test_medium_cloud_gives_grid_10(self):
        assert _adaptive_grid(100) == 10
        assert _adaptive_grid(5000) == 10
        assert _adaptive_grid(10_000) == 10

    def test_large_cloud_gives_grid_20(self):
        assert _adaptive_grid(10_001) == 20
        assert _adaptive_grid(100_000) == 20

    def test_small_cloud_total_voxels_is_125(self):
        score = coverage_from_point_cloud("r1", [(5.0, 5.0, 5.0)], _BBOX)
        # 1 point < 100 → GRID=5 → 125 voxels
        assert score.total_pixels == 125

    def test_large_cloud_total_voxels_is_8000(self):
        # 10001 points → GRID=20 → 8000 voxels
        # Use a grid pattern to fill the bbox deterministically
        many = [
            (float(i % 22) * 0.45 + 0.1, float((i // 22) % 22) * 0.45 + 0.1,
             float((i // 484) % 22) * 0.45 + 0.1)
            for i in range(10_001)
        ]
        score = coverage_from_point_cloud("r1", many, _BBOX)
        assert score.total_pixels == 8000


# ---------------------------------------------------------------------------
# pair_by_subject
# ---------------------------------------------------------------------------

class _MockRecord:
    """Minimal stand-in for DatasetRecord."""

    def __init__(self, record_id: str, notes: str = "") -> None:
        self.record_id = record_id
        self.notes = notes


def _make_phone_record(subject: str, timepoint: str, stem: str = "img") -> _MockRecord:
    record_id = f"phonecap_{subject}_{timepoint}_{stem}"
    notes = f"subject={subject} timepoint={timepoint}"
    return _MockRecord(record_id=record_id, notes=notes)


class TestPairBySubject:
    def test_empty_input_returns_empty_list(self):
        assert pair_by_subject([]) == []

    def test_single_timepoint_not_paired(self):
        records = [_make_phone_record("s001", "t0")]
        assert pair_by_subject(records) == []

    def test_two_timepoints_form_one_pair(self):
        records = [
            _make_phone_record("s001", "t0"),
            _make_phone_record("s001", "t1"),
        ]
        pairs = pair_by_subject(records)
        assert len(pairs) == 1
        assert pairs[0].subject_id == "s001"

    def test_t0_is_earliest_timepoint(self):
        records = [
            _make_phone_record("s001", "t2"),
            _make_phone_record("s001", "t0"),
            _make_phone_record("s001", "t1"),
        ]
        pairs = pair_by_subject(records)
        assert len(pairs) == 1
        # Earliest lexicographic timepoint should be t0.
        assert pairs[0].t0_record.notes.startswith("subject=s001 timepoint=t0")
        assert pairs[0].t1_record.notes.startswith("subject=s001 timepoint=t2")

    def test_t1_is_latest_timepoint(self):
        records = [
            _make_phone_record("s001", "t0"),
            _make_phone_record("s001", "t1"),
        ]
        pairs = pair_by_subject(records)
        assert pairs[0].t1_record.notes.startswith("subject=s001 timepoint=t1")

    def test_timepoints_natural_sorted_t10_after_t2(self):
        # Lexicographic ordering would put "t10" before "t2"; natural ordering
        # must treat t10 as the latest timepoint.
        records = [
            _make_phone_record("s001", "t2"),
            _make_phone_record("s001", "t10"),
            _make_phone_record("s001", "t1"),
        ]
        pairs = pair_by_subject(records)
        assert pairs[0].t0_record.notes.startswith("subject=s001 timepoint=t1")
        assert pairs[0].t1_record.notes.startswith("subject=s001 timepoint=t10")

    def test_non_numeric_timepoints_sort_lexicographically(self):
        # Labels with no digits fall back to lexicographic order.
        records = [
            _make_phone_record("s001", "followup"),
            _make_phone_record("s001", "baseline"),
        ]
        pairs = pair_by_subject(records)
        assert pairs[0].t0_record.notes.startswith("subject=s001 timepoint=baseline")
        assert pairs[0].t1_record.notes.startswith("subject=s001 timepoint=followup")

    def test_returns_longitudinal_pair_instances(self):
        records = [
            _make_phone_record("s001", "t0"),
            _make_phone_record("s001", "t1"),
        ]
        pairs = pair_by_subject(records)
        assert isinstance(pairs[0], LongitudinalPair)

    def test_multiple_subjects_form_multiple_pairs(self):
        records = [
            _make_phone_record("s001", "t0"),
            _make_phone_record("s001", "t1"),
            _make_phone_record("s002", "t0"),
            _make_phone_record("s002", "t1"),
        ]
        pairs = pair_by_subject(records)
        assert len(pairs) == 2
        subject_ids = {p.subject_id for p in pairs}
        assert subject_ids == {"s001", "s002"}

    def test_pairs_are_sorted_by_subject_id(self):
        records = [
            _make_phone_record("s002", "t0"),
            _make_phone_record("s002", "t1"),
            _make_phone_record("s001", "t0"),
            _make_phone_record("s001", "t1"),
        ]
        pairs = pair_by_subject(records)
        assert [p.subject_id for p in pairs] == ["s001", "s002"]

    def test_subject_with_no_parseable_id_is_skipped(self):
        # Record whose ID cannot be parsed by either strategy.
        bad_record = _MockRecord(record_id="no_prefix", notes="")
        good_records = [
            _make_phone_record("s001", "t0"),
            _make_phone_record("s001", "t1"),
        ]
        pairs = pair_by_subject([bad_record] + good_records)
        assert len(pairs) == 1

    def test_multiple_images_per_timepoint_uses_first(self):
        records = [
            _make_phone_record("s001", "t0", "img1"),
            _make_phone_record("s001", "t0", "img2"),
            _make_phone_record("s001", "t1", "img1"),
        ]
        pairs = pair_by_subject(records)
        assert len(pairs) == 1
        # t0 record should be one of the t0 images.
        assert "timepoint=t0" in pairs[0].t0_record.notes


class TestSubjectIdFromRecord:
    """Tests for _subject_id_from_record fallback paths (line 44 coverage)."""

    def test_phonecap_prefix_without_notes_returns_second_token(self):
        """Falls back to record_id token[1] when notes lacks 'subject=' (line 44)."""
        from dentalmapcert.longitudinal import _subject_id_from_record

        record = _MockRecord(record_id="phonecap_subj42_t0_anterior", notes="")
        assert _subject_id_from_record(record) == "subj42"

    def test_non_phonecap_prefix_returns_none(self):
        """record_id without 'phonecap' prefix → None (stays on return None path)."""
        from dentalmapcert.longitudinal import _subject_id_from_record

        record = _MockRecord(record_id="other_subj42_t0", notes="")
        assert _subject_id_from_record(record) is None


class TestTimepointFromRecord:
    """Tests for _timepoint_from_record fallback paths (lines 60-65 coverage)."""

    def test_record_id_fallback_returns_third_token(self):
        """When notes has no 'timepoint=', record_id with >=3 tokens returns token[2] (line 63)."""
        from dentalmapcert.longitudinal import _timepoint_from_record

        record = _MockRecord(record_id="phonecap_subj1_t1_anterior", notes="")
        assert _timepoint_from_record(record) == "t1"

    def test_record_id_too_short_returns_empty_string(self):
        """When record_id has <3 underscore tokens and no notes, returns '' (line 65)."""
        from dentalmapcert.longitudinal import _timepoint_from_record

        record = _MockRecord(record_id="short_id", notes="")
        assert _timepoint_from_record(record) == ""


class TestCoverageOpen3DExcept:
    """Tests for the Open3D exception fallback in coverage_from_point_cloud (lines 196-197)."""

    def test_open3d_exception_falls_through_to_raw_voxel_grid(self):
        """When open3d raises, the except block is hit and raw voxels are used (lines 196-197)."""
        import sys
        from unittest.mock import patch

        pts = _dense_grid(5)  # 125 points well inside bbox
        with patch.dict(sys.modules, {"open3d": None}):
            score = coverage_from_point_cloud("r1", pts, _BBOX)
        assert isinstance(score, CoverageScore)
        assert score.coverage_fraction > 0.0
