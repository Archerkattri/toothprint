"""Tests for scripts/failure_gallery.py."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from failure_gallery import (
    find_false_progressions,
    find_interval_failures,
    find_missed_detections,
    load_eval_rows,
    write_gallery,
)


class TestFindFalseProgressions(unittest.TestCase):
    def test_find_false_progressions(self):
        rows = [
            {"true": "stable", "decision": "progressed", "score": 5.0, "lo": 3.0, "hi": 7.0},
            {"true": "stable", "decision": "stable", "score": 1.0, "lo": 0.5, "hi": 2.0},
            {"true": "progressed", "decision": "progressed", "score": 8.0, "lo": 6.0, "hi": 10.0},
        ]
        result = find_false_progressions(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["score"], 5.0)
        self.assertEqual(result[0]["decision"], "progressed")
        self.assertEqual(result[0]["true"], "stable")

    def test_find_false_progressions_none(self):
        rows = [
            {"true": "stable", "decision": "stable"},
            {"true": "progressed", "decision": "progressed"},
        ]
        self.assertEqual(find_false_progressions(rows), [])


class TestFindMissedDetections(unittest.TestCase):
    def test_find_missed_detections(self):
        rows = [
            {"true": "progressed", "decision": "uncertain", "score": 4.0, "lo": 2.0, "hi": 6.0},
            {"true": "progressed", "decision": "progressed", "score": 9.0, "lo": 7.0, "hi": 11.0},
            {"true": "stable", "decision": "stable", "score": 1.0, "lo": 0.0, "hi": 3.0},
        ]
        result = find_missed_detections(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["score"], 4.0)
        self.assertEqual(result[0]["decision"], "uncertain")
        self.assertEqual(result[0]["true"], "progressed")

    def test_find_missed_detections_none(self):
        rows = [
            {"true": "progressed", "decision": "progressed"},
            {"true": "stable", "decision": "stable"},
        ]
        self.assertEqual(find_missed_detections(rows), [])


class TestFindIntervalFailures(unittest.TestCase):
    def test_find_interval_failures(self):
        rows = [
            # true_change=5.0 is outside [lo=6.0, hi=9.0] — failure
            {"true": "progressed", "decision": "uncertain", "score": 5.0,
             "lo": 6.0, "hi": 9.0, "true_change": 5.0},
            # true_change=0.0 (stable) is inside [lo=0.0, hi=2.0] — ok
            {"true": "stable", "decision": "stable", "score": 1.0,
             "lo": 0.0, "hi": 2.0},
        ]
        result = find_interval_failures(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["_true_change"], 5.0)

    def test_find_interval_failures_none_when_all_covered(self):
        rows = [
            {"true": "stable", "decision": "stable", "score": 1.0,
             "lo": 0.0, "hi": 2.0},
            {"true": "progressed", "decision": "progressed", "score": 8.0,
             "lo": 6.0, "hi": 10.0, "true_change": 8.0},
        ]
        result = find_interval_failures(rows)
        self.assertEqual(result, [])


class TestWriteGallery(unittest.TestCase):
    def test_write_gallery_creates_file(self):
        false_progs = [
            {"true": "stable", "decision": "progressed", "score": 5.0, "lo": 3.0, "hi": 7.0},
        ]
        missed = [
            {"true": "progressed", "decision": "uncertain", "score": 4.0, "lo": 2.0, "hi": 6.0},
        ]
        interval_fails = []

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "gallery"
            input_dir = Path("outputs/gate2_denpar")
            gallery_path = write_gallery(false_progs, missed, interval_fails, output_dir, input_dir)

            self.assertTrue(gallery_path.exists())
            self.assertEqual(gallery_path.name, "failure_gallery.md")
            content = gallery_path.read_text(encoding="utf-8")
            self.assertIn("# Failure-Case Gallery", content)
            self.assertIn("False progressions", content)
            self.assertIn("Missed Detections", content)
            self.assertIn("Interval Calibration Failures", content)

    def test_write_gallery_empty_categories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "gallery"
            gallery_path = write_gallery([], [], [], output_dir, Path("outputs/test"))
            content = gallery_path.read_text(encoding="utf-8")
            self.assertIn("_None._", content)


class TestLoadEvalRows(unittest.TestCase):
    def test_load_eval_rows_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_eval_rows(Path(tmpdir))
            self.assertEqual(result, [])

    def test_load_eval_rows_no_rows_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics_path.write_text(json.dumps({"false_progression_rate": 0.1}), encoding="utf-8")
            result = load_eval_rows(Path(tmpdir))
            self.assertEqual(result, [])

    def test_load_eval_rows_with_rows(self):
        rows = [{"true": "stable", "decision": "stable", "score": 1.0}]
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
            result = load_eval_rows(Path(tmpdir))
            self.assertEqual(result, rows)


if __name__ == "__main__":
    unittest.main()
