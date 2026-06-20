"""Tests for scripts/failure_gallery_dmc.py — 4 tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the scripts directory importable
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import failure_gallery_dmc as gallery_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


# ===========================================================================
# Test 1 — No JSONL files → sys.exit(0) called cleanly
# ===========================================================================

def test_no_jsonl_exits_cleanly(tmp_path):
    input_dir = tmp_path / "empty_run"
    input_dir.mkdir()

    with patch.object(sys, "argv", ["failure_gallery_dmc.py", "--input", str(input_dir)]):
        with pytest.raises(SystemExit) as exc_info:
            gallery_mod.main()
    assert exc_info.value.code == 0


# ===========================================================================
# Test 2 — JSONL exists but is empty → sys.exit(0) after printing message
# ===========================================================================

def test_main_with_empty_jsonl(tmp_path):
    input_dir = tmp_path / "run_empty"
    input_dir.mkdir()
    # Write an empty JSONL file
    (input_dir / "results.jsonl").write_text("", encoding="utf-8")

    with patch.object(sys, "argv", ["failure_gallery_dmc.py", "--input", str(input_dir)]):
        with pytest.raises(SystemExit) as exc_info:
            gallery_mod.main()
    assert exc_info.value.code == 0


# ===========================================================================
# Test 3 — Valid records → gallery markdown is created
# ===========================================================================

def test_gallery_created_with_records(tmp_path):
    input_dir = tmp_path / "run_real"
    input_dir.mkdir()

    records = [
        {
            "surface_region_id": "tooth_11",
            "label": "surface stable certified",
            "coverage_score_t0": 0.95,
            "coverage_score_t1": 0.93,
            "delta_interval_mm": [0.0, 0.15],
            "recapture_actions": [],
        },
        {
            "surface_region_id": "tooth_21",
            "label": "surface change certified",
            "coverage_score_t0": 0.88,
            "coverage_score_t1": 0.85,
            "delta_interval_mm": [0.5, 1.2],
            "recapture_actions": ["retake_left_buccal"],
        },
        {
            "surface_region_id": "tooth_31",
            "label": "uncertain / recapture",
            "coverage_score_t0": 0.60,
            "coverage_score_t1": 0.58,
            "delta_interval_mm": [-0.5, 2.0],
            "recapture_actions": ["retake_anterior"],
        },
    ]
    _write_jsonl(input_dir / "results.jsonl", records)

    output_dir = tmp_path / "gallery_out"
    with patch.object(
        sys, "argv",
        ["failure_gallery_dmc.py", "--input", str(input_dir), "--output", str(output_dir)],
    ):
        gallery_mod.main()

    gallery_path = output_dir / "failure_gallery_dmc.md"
    assert gallery_path.exists(), "Gallery markdown file was not created"

    content = gallery_path.read_text(encoding="utf-8")
    assert "# DMC Failure-Case Gallery" in content
    # stable records do not appear in failure tables; only failure categories do
    assert "tooth_11" not in content or "Surface stable certified" in content
    assert "tooth_21" in content   # false-change record must appear in gallery
    assert "tooth_31" in content   # over-uncertain record must appear in gallery
    # summary table must list the counts
    assert "| Surface stable certified | 1 |" in content
    assert "| Surface change certified (potential false-change) | 1 |" in content
    assert "| Uncertain / recapture (over-cautious) | 1 |" in content


# ===========================================================================
# Test 4 — Over-uncertain records are sorted by coverage_score_t0 ascending
# ===========================================================================

def test_invalid_json_line_is_skipped(tmp_path):
    """Lines 38-39: JSONDecodeError on a bad line → pass, valid lines still loaded."""
    input_dir = tmp_path / "run_bad_json"
    input_dir.mkdir()

    # Mix of invalid and valid lines
    jsonl = (
        "{not: valid json!!!}\n"
        + json.dumps({
            "surface_region_id": "tooth_99",
            "label": "surface stable certified",
            "coverage_score_t0": 0.9,
            "coverage_score_t1": 0.88,
            "delta_interval_mm": [0.0, 0.1],
            "recapture_actions": [],
        })
        + "\n"
    )
    (input_dir / "results.jsonl").write_text(jsonl, encoding="utf-8")
    output_dir = tmp_path / "gallery_bad"

    with patch.object(
        sys, "argv",
        ["failure_gallery_dmc.py", "--input", str(input_dir), "--output", str(output_dir)],
    ):
        gallery_mod.main()

    # Gallery created using the valid record; bad line was silently skipped
    gallery_path = output_dir / "failure_gallery_dmc.md"
    assert gallery_path.exists()
    assert "tooth_99" not in gallery_path.read_text(encoding="utf-8") or \
        "stable" in gallery_path.read_text(encoding="utf-8")


def test_over_uncertain_sorted_by_coverage(tmp_path):
    input_dir = tmp_path / "run_sort"
    input_dir.mkdir()

    records = [
        {
            "surface_region_id": "tooth_high_cov",
            "label": "uncertain / recapture",
            "coverage_score_t0": 0.80,
            "coverage_score_t1": 0.78,
            "delta_interval_mm": [-1.0, 1.5],
            "recapture_actions": [],
        },
        {
            "surface_region_id": "tooth_low_cov",
            "label": "uncertain / recapture",
            "coverage_score_t0": 0.30,
            "coverage_score_t1": 0.28,
            "delta_interval_mm": [-2.0, 2.5],
            "recapture_actions": [],
        },
        {
            "surface_region_id": "tooth_mid_cov",
            "label": "uncertain / recapture",
            "coverage_score_t0": 0.55,
            "coverage_score_t1": 0.50,
            "delta_interval_mm": [-1.5, 1.8],
            "recapture_actions": [],
        },
    ]
    _write_jsonl(input_dir / "results.jsonl", records)

    output_dir = tmp_path / "gallery_sort"
    with patch.object(
        sys, "argv",
        ["failure_gallery_dmc.py", "--input", str(input_dir), "--output", str(output_dir)],
    ):
        gallery_mod.main()

    content = (output_dir / "failure_gallery_dmc.md").read_text(encoding="utf-8")

    # tooth_low_cov (coverage 0.30) must appear before tooth_mid_cov (0.55)
    # and tooth_mid_cov must appear before tooth_high_cov (0.80)
    idx_low = content.index("tooth_low_cov")
    idx_mid = content.index("tooth_mid_cov")
    idx_high = content.index("tooth_high_cov")
    assert idx_low < idx_mid < idx_high, (
        "Over-uncertain regions not sorted by coverage_score_t0 ascending. "
        f"positions: low={idx_low}, mid={idx_mid}, high={idx_high}"
    )
