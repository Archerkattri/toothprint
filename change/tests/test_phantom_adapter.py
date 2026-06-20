"""Tests for dcc/data/phantom_adapter.py — 10 tests."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path


def _make_annotation(tooth_id: str = "1") -> dict:
    """Minimal DenPAR-style annotation dict."""
    return {
        "image": f"tooth_{tooth_id}.png",
        "teeth": [
            {
                "tooth_id": tooth_id,
                "cej": [[10.0, 20.0], [30.0, 20.5]],
                "crest_line": [[11.0, 35.0], [29.0, 35.5]],
            }
        ],
    }


def _write_acq(phantom_dir: Path, idx: int, ann: dict | None = None) -> None:
    if ann is None:
        ann = _make_annotation()
    (phantom_dir / f"acq_{idx:03d}.json").write_text(json.dumps(ann), encoding="utf-8")


class TestPhantomAdapterEmptyAndMissing(unittest.TestCase):

    def test_empty_root_yields_no_records(self):
        from dcc.data.phantom_adapter import PhantomAdapter

        with tempfile.TemporaryDirectory() as tmp:
            adapter = PhantomAdapter(tmp)
            self.assertEqual(adapter.records(), [])

    def test_nonexistent_root_yields_no_records(self):
        from dcc.data.phantom_adapter import PhantomAdapter

        adapter = PhantomAdapter("/nonexistent/path/that/does/not/exist_xyz")
        self.assertEqual(adapter.records(), [])


class TestPhantomAdapterSinglePhantom(unittest.TestCase):

    def test_single_phantom_single_acquisition(self):
        from dcc.data.phantom_adapter import PhantomAdapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_dir = root / "phantom_001"
            p_dir.mkdir()
            _write_acq(p_dir, 0)

            records = PhantomAdapter(root).records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].phantom_id, "phantom_001")
            self.assertEqual(records[0].acquisition_idx, 0)
            self.assertEqual(records[0].image_id, "phantom_001_acq_000")

    def test_single_phantom_multiple_acquisitions(self):
        from dcc.data.phantom_adapter import PhantomAdapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_dir = root / "phantom_001"
            p_dir.mkdir()
            for i in range(4):
                _write_acq(p_dir, i)

            records = PhantomAdapter(root).records()
            self.assertEqual(len(records), 4)
            for i, rec in enumerate(records):
                self.assertEqual(rec.acquisition_idx, i)


class TestPhantomAdapterStablePairs(unittest.TestCase):

    def test_stable_pairs_count(self):
        """3 acquisitions → C(3,2) = 3 pairs."""
        from dcc.data.phantom_adapter import PhantomAdapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_dir = root / "phantom_001"
            p_dir.mkdir()
            for i in range(3):
                _write_acq(p_dir, i)

            pairs = PhantomAdapter(root).stable_pairs()
            self.assertEqual(len(pairs), 3)

    def test_all_pairs_labelled_stable(self):
        from dcc.data.phantom_adapter import PhantomAdapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_dir = root / "phantom_001"
            p_dir.mkdir()
            for i in range(3):
                _write_acq(p_dir, i)

            pairs = PhantomAdapter(root).stable_pairs()
            for pair in pairs:
                self.assertEqual(pair["label"], "stable")

    def test_true_change_is_zero_for_all_pairs(self):
        from dcc.data.phantom_adapter import PhantomAdapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_dir = root / "phantom_001"
            p_dir.mkdir()
            for i in range(3):
                _write_acq(p_dir, i)

            pairs = PhantomAdapter(root).stable_pairs()
            for pair in pairs:
                self.assertEqual(pair["true_change"], 0.0)


class TestPhantomAdapterMetadata(unittest.TestCase):

    def test_material_loaded_from_metadata_json(self):
        from dcc.data.phantom_adapter import PhantomAdapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_dir = root / "phantom_001"
            p_dir.mkdir()
            _write_acq(p_dir, 0)
            (p_dir / "metadata.json").write_text(
                json.dumps({"material": "resin"}), encoding="utf-8"
            )

            records = PhantomAdapter(root).records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].material, "resin")

    def test_missing_metadata_defaults_to_unknown(self):
        from dcc.data.phantom_adapter import PhantomAdapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_dir = root / "phantom_001"
            p_dir.mkdir()
            _write_acq(p_dir, 0)

            records = PhantomAdapter(root).records()
            self.assertEqual(records[0].material, "unknown")


class TestLoadPhantomDataConvenience(unittest.TestCase):

    def test_load_phantom_data_convenience_function(self):
        from dcc.data.phantom_adapter import load_phantom_data

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_dir = root / "phantom_A"
            p_dir.mkdir()
            _write_acq(p_dir, 0)
            _write_acq(p_dir, 1)

            records = load_phantom_data(root)
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r.phantom_id == "phantom_A" for r in records))


class TestPhantomAdapterEdgeCases(unittest.TestCase):

    def test_non_dir_item_in_root_is_skipped(self):
        """Line 49: items in root that are not directories are skipped."""
        from dcc.data.phantom_adapter import PhantomAdapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a_plain_file.txt").write_text("not a dir")
            p_dir = root / "phantom_001"
            p_dir.mkdir()
            _write_acq(p_dir, 0)

            records = PhantomAdapter(root).records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].phantom_id, "phantom_001")

    def test_invalid_metadata_json_is_silently_ignored(self):
        """Lines 58-59: invalid metadata.json → except Exception: pass → material='unknown'."""
        from dcc.data.phantom_adapter import PhantomAdapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_dir = root / "phantom_001"
            p_dir.mkdir()
            _write_acq(p_dir, 0)
            (p_dir / "metadata.json").write_text("{not: valid json!!!}", encoding="utf-8")

            records = PhantomAdapter(root).records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].material, "unknown")

    def test_invalid_acquisition_json_is_skipped(self):
        """Lines 65-66: invalid acq_*.json → except Exception: continue → record skipped."""
        from dcc.data.phantom_adapter import PhantomAdapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_dir = root / "phantom_001"
            p_dir.mkdir()
            _write_acq(p_dir, 0)
            (p_dir / "acq_001.json").write_text("{not: valid json!!!}", encoding="utf-8")

            records = PhantomAdapter(root).records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].acquisition_idx, 0)


if __name__ == "__main__":
    unittest.main()
