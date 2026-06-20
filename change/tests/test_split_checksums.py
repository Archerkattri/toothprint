"""Tests for dcc.splits.checksums.

The new ``compute_record_checksum(image_path, label_path)`` hashes actual
file bytes, so tests that exercise it must create real temporary files via
``tmp_path`` (pytest) or ``tempfile.TemporaryDirectory`` (unittest).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Minimal fake record (with real file support)
# ---------------------------------------------------------------------------

@dataclass
class _Rec:
    image_id: str
    image_path: Path
    label_path: Path | None = None
    split: str = "baseline"
    annotation_dict: dict = field(default_factory=dict)

    def __post_init__(self):
        # If label_path not set, fall back to image_path (same file used for both)
        if self.label_path is None:
            self.label_path = self.image_path


def _make_records_with_files(n: int, tmpdir: Path, split: str = "baseline") -> list[_Rec]:
    """Create n _Rec objects backed by real files with unique content."""
    records = []
    for i in range(n):
        img_path = tmpdir / f"img{i:03d}.png"
        lbl_path = tmpdir / f"img{i:03d}.txt"
        # Unique content per file so checksums differ
        img_path.write_bytes(f"image_content_{i}".encode())
        lbl_path.write_bytes(f"label_content_{i}".encode())
        records.append(_Rec(
            image_id=f"img{i:03d}",
            image_path=img_path,
            label_path=lbl_path,
            split=split,
        ))
    return records


# ---------------------------------------------------------------------------
# compute_record_checksum
# ---------------------------------------------------------------------------

class TestComputeRecordChecksum(unittest.TestCase):
    def test_returns_hex_string(self):
        from dcc.splits.checksums import compute_record_checksum
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Path(tmpdir) / "img.png"
            lbl = Path(tmpdir) / "img.txt"
            img.write_bytes(b"image-bytes")
            lbl.write_bytes(b"label-bytes")
            cs = compute_record_checksum(img, lbl)
        self.assertIsInstance(cs, str)
        # SHA-256 hex = 64 chars
        self.assertEqual(len(cs), 64)
        int(cs, 16)  # must be valid hex

    def test_deterministic_for_same_files(self):
        from dcc.splits.checksums import compute_record_checksum
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Path(tmpdir) / "img.png"
            lbl = Path(tmpdir) / "img.txt"
            img.write_bytes(b"stable-image")
            lbl.write_bytes(b"stable-label")
            cs1 = compute_record_checksum(img, lbl)
            cs2 = compute_record_checksum(img, lbl)
        self.assertEqual(cs1, cs2)

    def test_different_file_content_produces_different_checksums(self):
        from dcc.splits.checksums import compute_record_checksum
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records_with_files(5, Path(tmpdir))
            checksums = [compute_record_checksum(r.image_path, r.label_path) for r in records]
        self.assertEqual(len(checksums), len(set(checksums)), "All checksums should be unique")

    def test_changed_image_content_changes_checksum(self):
        """Changing image file bytes changes the checksum."""
        from dcc.splits.checksums import compute_record_checksum
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Path(tmpdir) / "img.png"
            lbl = Path(tmpdir) / "img.txt"
            lbl.write_bytes(b"same-label")

            img.write_bytes(b"original-image-bytes")
            cs_original = compute_record_checksum(img, lbl)

            img.write_bytes(b"modified-image-bytes")
            cs_modified = compute_record_checksum(img, lbl)

        self.assertNotEqual(cs_original, cs_modified)

    def test_changed_label_content_changes_checksum(self):
        """Changing label file bytes changes the checksum."""
        from dcc.splits.checksums import compute_record_checksum
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Path(tmpdir) / "img.png"
            lbl = Path(tmpdir) / "img.txt"
            img.write_bytes(b"same-image")

            lbl.write_bytes(b"original-label")
            cs_original = compute_record_checksum(img, lbl)

            lbl.write_bytes(b"modified-label")
            cs_modified = compute_record_checksum(img, lbl)

        self.assertNotEqual(cs_original, cs_modified)

    def test_same_content_same_checksum_regardless_of_filename(self):
        """Content-based hash: same bytes → same digest even in different files."""
        from dcc.splits.checksums import compute_record_checksum
        with tempfile.TemporaryDirectory() as tmpdir:
            img_a = Path(tmpdir) / "fileA.png"
            img_b = Path(tmpdir) / "fileB.png"
            lbl = Path(tmpdir) / "shared.txt"
            img_a.write_bytes(b"identical-content")
            img_b.write_bytes(b"identical-content")
            lbl.write_bytes(b"identical-label")
            cs_a = compute_record_checksum(img_a, lbl)
            cs_b = compute_record_checksum(img_b, lbl)
        self.assertEqual(cs_a, cs_b)


# ---------------------------------------------------------------------------
# check_split_leakage
# ---------------------------------------------------------------------------

@dataclass
class _SimpleRec:
    image_id: str
    image_path: Path = field(default_factory=lambda: Path("/fake.png"))


class TestCheckSplitLeakage(unittest.TestCase):
    def test_no_overlap_returns_empty(self):
        from dcc.splits.checksums import check_split_leakage
        train = [_SimpleRec(image_id=f"train{i}") for i in range(3)]
        test = [_SimpleRec(image_id=f"test{i}") for i in range(3)]
        self.assertEqual(check_split_leakage(train, test), [])

    def test_complete_overlap_returns_all(self):
        from dcc.splits.checksums import check_split_leakage
        records = [_SimpleRec(image_id=f"img{i}") for i in range(4)]
        leaked = check_split_leakage(records, records)
        self.assertEqual(sorted(leaked), sorted(r.image_id for r in records))

    def test_partial_overlap_detected(self):
        from dcc.splits.checksums import check_split_leakage
        train = [_SimpleRec(image_id=x) for x in ["a", "b", "c"]]
        test = [_SimpleRec(image_id=x) for x in ["b", "d"]]
        self.assertEqual(check_split_leakage(train, test), ["b"])

    def test_returns_sorted_list(self):
        from dcc.splits.checksums import check_split_leakage
        train = [_SimpleRec(image_id=x) for x in ["c", "a", "b"]]
        test = [_SimpleRec(image_id=x) for x in ["b", "a"]]
        self.assertEqual(check_split_leakage(train, test), ["a", "b"])

    def test_empty_splits_return_empty(self):
        from dcc.splits.checksums import check_split_leakage
        self.assertEqual(check_split_leakage([], []), [])
        self.assertEqual(check_split_leakage([_SimpleRec(image_id="x")], []), [])
        self.assertEqual(check_split_leakage([], [_SimpleRec(image_id="x")]), [])


# ---------------------------------------------------------------------------
# freeze_split_checksums
# ---------------------------------------------------------------------------

class TestFreezeSplitChecksums(unittest.TestCase):
    def test_creates_valid_json(self):
        from dcc.splits.checksums import freeze_split_checksums
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records_with_files(4, Path(tmpdir))
            out_path = Path(tmpdir) / "checksums.json"
            freeze_split_checksums(records, out_path)
            payload = json.loads(out_path.read_text())
        self.assertIn("schema_version", payload)
        self.assertIn("records", payload)
        self.assertEqual(len(payload["records"]), 4)

    def test_records_sorted_by_image_id(self):
        from dcc.splits.checksums import freeze_split_checksums
        with tempfile.TemporaryDirectory() as tmpdir:
            z = Path(tmpdir) / "z.png"
            a = Path(tmpdir) / "a.png"
            m = Path(tmpdir) / "m.png"
            for p in [z, a, m]:
                p.write_bytes(p.name.encode())
            records = [
                _Rec(image_id="z", image_path=z),
                _Rec(image_id="a", image_path=a),
                _Rec(image_id="m", image_path=m),
            ]
            out_path = Path(tmpdir) / "cs.json"
            freeze_split_checksums(records, out_path)
            payload = json.loads(out_path.read_text())
        ids = [e["image_id"] for e in payload["records"]]
        self.assertEqual(ids, sorted(ids))

    def test_each_entry_has_checksum_field(self):
        from dcc.splits.checksums import freeze_split_checksums
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records_with_files(3, Path(tmpdir))
            out = Path(tmpdir) / "cs.json"
            freeze_split_checksums(records, out)
            payload = json.loads(out.read_text())
        for entry in payload["records"]:
            self.assertIn("checksum", entry)
            self.assertEqual(len(entry["checksum"]), 64)

    def test_creates_parent_directories(self):
        from dcc.splits.checksums import freeze_split_checksums
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records_with_files(2, Path(tmpdir))
            nested = Path(tmpdir) / "a" / "b" / "checksums.json"
            freeze_split_checksums(records, nested)
            self.assertTrue(nested.exists())

    def test_returns_path(self):
        from dcc.splits.checksums import freeze_split_checksums
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records_with_files(2, Path(tmpdir))
            out = Path(tmpdir) / "cs.json"
            result = freeze_split_checksums(records, out)
            self.assertEqual(Path(result), out)

    def test_freeze_checksums_without_label_path_emits_warning(self):
        """freeze_split_checksums emits UserWarning for records with no label_path (lines 103-109)."""
        import warnings
        from dcc.splits.checksums import freeze_split_checksums

        class _MinimalRec:
            image_id = "img0"
            image_path = None  # set below; intentionally NO label_path attribute

        with tempfile.TemporaryDirectory() as tmpdir:
            img = Path(tmpdir) / "img0.png"
            img.write_bytes(b"fake-image-content")
            rec = _MinimalRec()
            rec.image_path = img

            out = Path(tmpdir) / "checksums.json"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                freeze_split_checksums([rec], out)

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(
            any("label_path" in str(w.message) for w in user_warnings),
            f"Expected UserWarning mentioning label_path; got: {[str(w.message) for w in user_warnings]}",
        )


# ---------------------------------------------------------------------------
# verify_split_checksums
# ---------------------------------------------------------------------------

class TestVerifySplitChecksums(unittest.TestCase):
    def _write_and_verify(self, records: list, verify_records: list, tmpdir: Path) -> bool:
        from dcc.splits.checksums import freeze_split_checksums, verify_split_checksums
        cs_path = tmpdir / "cs.json"
        freeze_split_checksums(records, cs_path)
        return verify_split_checksums(verify_records, cs_path)

    def test_same_records_verifies_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records_with_files(5, Path(tmpdir))
            result = self._write_and_verify(records, records, Path(tmpdir))
        self.assertTrue(result)

    def test_missing_record_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records_with_files(5, Path(tmpdir))
            partial = records[:3]
            result = self._write_and_verify(records, partial, Path(tmpdir))
        self.assertFalse(result)

    def test_extra_record_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records_with_files(3, Path(tmpdir))
            extra_img = Path(tmpdir) / "new.png"
            extra_img.write_bytes(b"extra-image")
            extra = records + [_Rec(image_id="new", image_path=extra_img)]
            result = self._write_and_verify(records, extra, Path(tmpdir))
        self.assertFalse(result)

    def test_changed_file_content_fails(self):
        """Changing the image file bytes after freezing causes verify to return False."""
        from dcc.splits.checksums import freeze_split_checksums, verify_split_checksums
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Path(tmpdir) / "img.png"
            lbl = Path(tmpdir) / "img.txt"
            img.write_bytes(b"original-image")
            lbl.write_bytes(b"original-label")
            record = _Rec(image_id="img0", image_path=img, label_path=lbl)
            cs_path = Path(tmpdir) / "cs.json"
            freeze_split_checksums([record], cs_path)

            # Modify image content after freezing
            img.write_bytes(b"tampered-image")
            result = verify_split_checksums([record], cs_path)
        self.assertFalse(result)

    def test_empty_records_against_empty_file(self):
        from dcc.splits.checksums import freeze_split_checksums, verify_split_checksums
        with tempfile.TemporaryDirectory() as tmpdir:
            cs_path = Path(tmpdir) / "cs.json"
            freeze_split_checksums([], cs_path)
            result = verify_split_checksums([], cs_path)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
