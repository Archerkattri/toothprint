"""Tests for dcc.data.periapical_lesions_adapter."""

from __future__ import annotations

import unittest
from pathlib import Path

import pytest

DATASET_ROOT = Path(__file__).parent.parent / "data" / "periapical-lesions" / "extracted"


@pytest.fixture
def real_root():
    if not DATASET_ROOT.exists():
        pytest.skip("Periapical lesions dataset not found")
    return DATASET_ROOT


class TestPeriapicalLesionsAdapterSynthetic(unittest.TestCase):
    def test_empty_root_yields_no_records(self):
        from dcc.data.periapical_lesions_adapter import PeriapicalLesionsAdapter
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = PeriapicalLesionsAdapter(tmpdir)
            records = list(adapter.records())
        self.assertEqual(records, [])

    def test_load_periapical_lesions_convenience_function(self):
        from dcc.data.periapical_lesions_adapter import load_periapical_lesions
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_periapical_lesions(tmpdir)
        self.assertIsInstance(result, list)


class TestPeriapicalLesionsAdapterReal(unittest.TestCase):
    def setUp(self):
        if not DATASET_ROOT.exists():
            self.skipTest("Periapical lesions dataset not found")
        from dcc.data.periapical_lesions_adapter import PeriapicalLesionsAdapter
        self.adapter = PeriapicalLesionsAdapter(DATASET_ROOT)

    def test_yields_around_450_records(self):
        records = list(self.adapter.records())
        self.assertGreater(len(records), 400)
        self.assertLess(len(records), 500)

    def test_three_cameras_represented(self):
        cameras = {r.camera for r in self.adapter.records()}
        self.assertEqual(cameras, {"canon", "iphone", "xiaomi"})

    def test_all_records_have_image_path(self):
        for rec in self.adapter.records():
            self.assertIsInstance(rec.image_path, Path)
            self.assertTrue(rec.image_path.exists(), f"Missing: {rec.image_path}")

    def test_lesion_labels_are_bool(self):
        for rec in self.adapter.records():
            self.assertIsInstance(rec.lesion_present, bool)

    def test_lesion_rate_is_reasonable(self):
        records = list(self.adapter.records())
        lesion_count = sum(1 for r in records if r.lesion_present)
        rate = lesion_count / len(records)
        # DenPAR: ~40% lesion rate is expected
        self.assertGreater(rate, 0.2)
        self.assertLess(rate, 0.8)

    def test_splits_are_valid(self):
        valid_splits = {"train", "val", "test"}
        for rec in self.adapter.records():
            self.assertIn(rec.split, valid_splits)

    def test_split_filter_works(self):
        all_recs = list(self.adapter.records())
        train = list(self.adapter.records(split="train"))
        val = list(self.adapter.records(split="val"))
        test = list(self.adapter.records(split="test"))
        self.assertEqual(len(train) + len(val) + len(test), len(all_recs))

    def test_camera_filter_works(self):
        all_recs = list(self.adapter.records())
        canon = list(self.adapter.records(camera="canon"))
        iphone = list(self.adapter.records(camera="iphone"))
        xiaomi = list(self.adapter.records(camera="xiaomi"))
        self.assertEqual(len(canon) + len(iphone) + len(xiaomi), len(all_recs))

    def test_rx_numbers_are_positive_ints(self):
        for rec in self.adapter.records():
            self.assertIsInstance(rec.rx_number, int)
            self.assertGreater(rec.rx_number, 0)

    def test_image_ids_are_unique(self):
        records = list(self.adapter.records())
        ids = [r.image_id for r in records]
        self.assertEqual(len(ids), len(set(ids)))


class TestPeriapicalAdapterSyntheticEdgeCases(unittest.TestCase):
    """Cover missing lines 99-100, 124, 128 in periapical_lesions_adapter.py."""

    def test_openpyxl_import_error_returns_empty_labels(self):
        """Lines 99-100: when openpyxl is not importable, _load_labels returns {}."""
        import sys
        import tempfile
        from unittest.mock import patch
        from dcc.data.periapical_lesions_adapter import PeriapicalLesionsAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create the xlsx file so the code tries to import openpyxl
            (root / "periapical_lesions_classification.xlsx").write_bytes(b"fake-xlsx")
            adapter = PeriapicalLesionsAdapter(root)
            with patch.dict(sys.modules, {'openpyxl': None}):
                labels = adapter._load_labels()
        self.assertEqual(labels, {})

    def test_non_image_suffix_in_camera_dir_is_skipped(self):
        """Line 124: files with non-image suffix inside camera dir are skipped."""
        import tempfile
        from dcc.data.periapical_lesions_adapter import PeriapicalLesionsAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Real camera dir name from _CAMERA_DIRS: "1. Rx Canon", prefix "1."
            cam_dir = root / "1. Rx Canon"
            cam_dir.mkdir()
            (cam_dir / "1.5.txt").write_bytes(b"not an image")

            adapter = PeriapicalLesionsAdapter(root)
            records = list(adapter.records())
        self.assertEqual(records, [])

    def test_wrong_prefix_stem_in_camera_dir_is_skipped(self):
        """Line 128: file whose stem doesn't start with the expected prefix is skipped."""
        import tempfile
        from dcc.data.periapical_lesions_adapter import PeriapicalLesionsAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cam_dir = root / "1. Rx Canon"
            cam_dir.mkdir()
            # Canon dir expects prefix "1." but this file uses "X."
            (cam_dir / "X.37.JPG").write_bytes(b"\xff\xd8" + b"\x00" * 10)

            adapter = PeriapicalLesionsAdapter(root)
            records = list(adapter.records())
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
