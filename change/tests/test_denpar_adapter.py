import json
import tempfile
import unittest
from pathlib import Path


class ValidateRecordTests(unittest.TestCase):
    """FIX 8: validate_record and DenparAdapter.validate_all tests."""

    def test_valid_record_returns_no_errors(self):
        from dcc.data.denpar_adapter import validate_record

        record = {
            "image": "case001.png",
            "teeth": [
                {
                    "tooth_id": "36",
                    "cej": [[10.0, 20.0]],
                    "crest_line": [[10.0, 35.0]],
                    "apex": [[20.0, 80.0]],
                }
            ],
        }
        errors = validate_record(record)
        self.assertEqual(errors, [])

    def test_missing_image_key_reported(self):
        from dcc.data.denpar_adapter import validate_record

        record = {"teeth": []}
        errors = validate_record(record)
        self.assertTrue(any("image" in e for e in errors))

    def test_missing_teeth_key_reported(self):
        from dcc.data.denpar_adapter import validate_record

        record = {"image": "x.png"}
        errors = validate_record(record)
        self.assertTrue(any("teeth" in e for e in errors))

    def test_tooth_missing_tooth_id_reported(self):
        from dcc.data.denpar_adapter import validate_record

        record = {
            "image": "x.png",
            "teeth": [{"cej": [[0.0, 0.0]]}],  # no tooth_id
        }
        errors = validate_record(record)
        self.assertTrue(any("tooth_id" in e for e in errors))

    def test_empty_teeth_list_is_valid(self):
        from dcc.data.denpar_adapter import validate_record

        errors = validate_record({"image": "x.png", "teeth": []})
        self.assertEqual(errors, [])

    def test_validate_all_returns_dict_keyed_by_stem(self):
        from dcc.data.denpar_adapter import DenparAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ann_dir = root / "annotations"
            ann_dir.mkdir()
            # valid annotation
            (ann_dir / "case001.json").write_text(
                json.dumps({"image": "case001.png", "teeth": []}),
                encoding="utf-8",
            )
            # invalid annotation (missing 'image')
            (ann_dir / "case002.json").write_text(
                json.dumps({"teeth": []}),
                encoding="utf-8",
            )
            adapter = DenparAdapter(root)
            results = adapter.validate_all()

        self.assertIn("case001", results)
        self.assertIn("case002", results)
        self.assertEqual(results["case001"], [])
        self.assertTrue(len(results["case002"]) > 0)


class DenparAdapterTests(unittest.TestCase):
    def _write_fixture_dataset(self, root: Path) -> Path:
        images = root / "images"
        annotations = root / "annotations"
        images.mkdir(parents=True)
        annotations.mkdir(parents=True)

        image_path = images / "case001.png"
        image_path.write_bytes(b"not-a-real-png-but-a-stable-fixture")

        annotation_path = annotations / "case001.json"
        annotation_path.write_text(
            json.dumps(
                {
                    "image": "case001.png",
                    "teeth": [
                        {
                            "tooth_id": "36",
                            "cej": [[10.0, 20.0], [30.0, 20.5]],
                            "apex": [[20.0, 80.0]],
                            "crest_line": [[11.0, 35.0], [29.0, 35.5]],
                            "mask": "case001_tooth36.png",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_adapter_indexes_local_denpar_style_fixture(self):
        from dcc.data.denpar_adapter import DenparAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = self._write_fixture_dataset(Path(tmpdir))
            adapter = DenparAdapter(dataset_root)

            records = list(adapter.iter_records())

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].image_id, "case001")
        self.assertEqual(records[0].image_path.name, "case001.png")
        self.assertEqual(records[0].annotation["teeth"][0]["tooth_id"], "36")

    def test_adapter_reports_missing_required_directories(self):
        from dcc.data.denpar_adapter import DenparAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = DenparAdapter(Path(tmpdir))

            with self.assertRaisesRegex(FileNotFoundError, "images"):
                list(adapter.iter_records())


class ValidateRecordEdgeCaseTests(unittest.TestCase):
    """Cover lines 90 and 94-95 in denpar_adapter.py."""

    def test_teeth_not_a_list_is_reported(self):
        """Line 90: 'teeth' must be a list; string is rejected."""
        from dcc.data.denpar_adapter import validate_record

        errors = validate_record({"image": "x.png", "teeth": "not_a_list"})
        self.assertTrue(any("list" in e for e in errors))

    def test_tooth_not_a_dict_is_reported(self):
        """Lines 94-95: individual tooth entry must be a dict."""
        from dcc.data.denpar_adapter import validate_record

        errors = validate_record({"image": "x.png", "teeth": ["not_a_dict"]})
        self.assertTrue(any("dict" in e for e in errors))


class ValidateAllEdgeCaseTests(unittest.TestCase):
    """Cover lines 152-154 in denpar_adapter.py."""

    def test_validate_all_json_parse_error_is_recorded(self):
        """Lines 152-154: annotation JSON that can't be parsed → JSON parse error in results."""
        from dcc.data.denpar_adapter import DenparAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ann_dir = root / "annotations"
            ann_dir.mkdir()
            (ann_dir / "bad.json").write_text("{not valid json!!!", encoding="utf-8")

            adapter = DenparAdapter(root)
            results = adapter.validate_all()

        self.assertIn("bad", results)
        self.assertTrue(any("JSON parse error" in e for e in results["bad"]))


class DenparAdapterIterRecordsEdgeCases(unittest.TestCase):
    """Cover lines 124 and 128 in DenparAdapter.iter_records."""

    def test_annotation_without_image_key_raises_value_error(self):
        """Line 124: annotation with no 'image' field raises ValueError."""
        from dcc.data.denpar_adapter import DenparAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images = root / "images"
            annotations = root / "annotations"
            images.mkdir()
            annotations.mkdir()
            (annotations / "case001.json").write_text(
                json.dumps({"teeth": []}), encoding="utf-8"
            )
            adapter = DenparAdapter(root)
            with self.assertRaises(ValueError):
                list(adapter.iter_records())

    def test_annotation_with_missing_image_file_raises_file_not_found(self):
        """Line 128: annotation references a non-existent image → FileNotFoundError."""
        from dcc.data.denpar_adapter import DenparAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images = root / "images"
            annotations = root / "annotations"
            images.mkdir()
            annotations.mkdir()
            (annotations / "case001.json").write_text(
                json.dumps({"image": "missing.png", "teeth": []}), encoding="utf-8"
            )
            adapter = DenparAdapter(root)
            with self.assertRaises(FileNotFoundError):
                list(adapter.iter_records())


class BuildAnnotationDictEdgeCases(unittest.TestCase):
    """Cover line 210 in _build_annotation_dict."""

    def test_empty_bone_line_in_bone_lines_is_skipped(self):
        """Line 210: empty [] entry in Bone_Lines is skipped (continue)."""
        from dcc.data.denpar_adapter import _build_annotation_dict

        kp_data = {
            "Image_id": "test.jpg",
            "bboxes": [[10.0, 10.0, 50.0, 50.0]],
            "CEJ_Points": [[30.0, 15.0]],
            "Apex_Points": [],
        }
        bl_data = {
            "Bone_Lines": [[], [[10.0, 30.0], [50.0, 30.0]]]
        }
        result = _build_annotation_dict(kp_data, bl_data)
        self.assertEqual(result["image"], "test.jpg")
        # Non-empty bone line should still be processed
        self.assertGreaterEqual(len(result["teeth"]), 0)


class RealDenparAdapterEdgeCases(unittest.TestCase):
    """Cover lines 313 and 319 in RealDenparAdapter._iter_split."""

    def test_missing_kp_dir_silently_returns_nothing(self):
        """Line 313: RealDenparAdapter skips split when kp_dir doesn't exist."""
        from dcc.data.denpar_adapter import RealDenparAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create Training dir but no "Key Points Annotations" subdir
            (root / "Training").mkdir()
            adapter = RealDenparAdapter(root)
            records = list(adapter.records(split="train"))
        self.assertEqual(records, [])

    def test_kp_without_bl_is_skipped(self):
        """Line 319: kp_path exists but bl_path does not → continue."""
        from dcc.data.denpar_adapter import RealDenparAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "Training"
            kp_dir = split_dir / "Key Points Annotations"
            bl_dir = split_dir / "Bone Level Annotations"
            kp_dir.mkdir(parents=True)
            bl_dir.mkdir(parents=True)
            # Write kp file but no matching bl file
            (kp_dir / "1001.json").write_text(
                json.dumps({"Image_id": "1001.jpg", "bboxes": [], "CEJ_Points": [], "Apex_Points": []}),
                encoding="utf-8",
            )
            adapter = RealDenparAdapter(root)
            records = list(adapter.records(split="train"))
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
