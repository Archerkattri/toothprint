"""Tests for dcc.data.image_pair_builder."""

from __future__ import annotations

import struct
import tempfile
import unittest
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_png(path: Path, width: int = 10, height: int = 8) -> Path:
    """Write a minimal valid RGB PNG with the given dimensions."""
    def _u32be(n: int) -> bytes:
        return struct.pack(">I", n)

    def _chunk(name: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return _u32be(len(data)) + name + data + _u32be(crc)

    ihdr_data = (
        _u32be(width)
        + _u32be(height)
        + bytes([8, 2, 0, 0, 0])  # bit_depth=8, color_type=2 (RGB)
    )
    raw_row = bytes([0]) + bytes([128, 64, 32] * width)  # filter byte + grey-ish RGB
    raw = raw_row * height
    idat_data = zlib.compress(raw)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr_data)
        + _chunk(b"IDAT", idat_data)
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return path


def _make_annotation(tooth_id: str = "1") -> dict:
    return {
        "teeth": [
            {
                "tooth_id": tooth_id,
                "cej": [[2.0, 3.0], [4.0, 3.0]],
                "crest_line": [[2.0, 5.0], [4.0, 5.0]],
                "apex": [[3.0, 8.0]],
            }
        ]
    }


@dataclass
class _FakeRecord:
    image_id: str
    image_path: Path
    annotation_dict: dict
    split: str = "baseline"


def _make_records(n: int, tmpdir: Path) -> list[_FakeRecord]:
    records = []
    for i in range(n):
        img_path = tmpdir / f"img{i}.png"
        _make_png(img_path)
        records.append(_FakeRecord(
            image_id=f"img{i}",
            image_path=img_path,
            annotation_dict=_make_annotation(str(i + 1)),
        ))
    return records


# ---------------------------------------------------------------------------
# Tests for image loading
# ---------------------------------------------------------------------------

class TestLoadImage(unittest.TestCase):
    def test_load_png_returns_array(self):
        from dcc.data.image_pair_builder import _load_image
        with tempfile.TemporaryDirectory() as tmpdir:
            p = _make_png(Path(tmpdir) / "test.png", width=10, height=8)
            img = _load_image(p)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (8, 10, 3))

    def test_load_png_values_in_0_1(self):
        from dcc.data.image_pair_builder import _load_image
        with tempfile.TemporaryDirectory() as tmpdir:
            p = _make_png(Path(tmpdir) / "test.png")
            img = _load_image(p)
        lo, hi = float(img.min()), float(img.max())
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)


# ---------------------------------------------------------------------------
# Tests for _perturb_to_uint8
# ---------------------------------------------------------------------------

class TestPerturbToUint8(unittest.TestCase):
    def _make_float_image(self, h: int = 4, w: int = 6):
        return np.full((h, w, 3), 0.5, dtype=np.float32)

    def test_output_values_are_uint8_range(self):
        from dcc.data.image_pair_builder import _perturb_to_uint8
        from dcc.perturb.image_perturb import ImagePerturbConfig
        img = self._make_float_image()
        out = _perturb_to_uint8(img, ImagePerturbConfig())
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreaterEqual(int(out.min()), 0)
        self.assertLessEqual(int(out.max()), 255)

    def test_neutral_config_gives_near_128(self):
        """0.5 float × 255 = 127 or 128 depending on rounding."""
        from dcc.data.image_pair_builder import _perturb_to_uint8
        from dcc.perturb.image_perturb import ImagePerturbConfig
        img = self._make_float_image()
        out = _perturb_to_uint8(img, ImagePerturbConfig())
        val = int(out[0, 0, 0])
        self.assertIn(val, range(127, 130))


# ---------------------------------------------------------------------------
# Tests for build_image_pairs
# ---------------------------------------------------------------------------

class TestBuildImagePairs(unittest.TestCase):
    def test_two_pairs_per_record(self):
        from dcc.data.image_pair_builder import build_image_pairs
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records(3, Path(tmpdir))
            pairs = build_image_pairs(records)
        self.assertEqual(len(pairs), 6)  # 3 records × 2 pairs each

    def test_labels_stable_then_progressed(self):
        from dcc.data.image_pair_builder import build_image_pairs
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records(2, Path(tmpdir))
            pairs = build_image_pairs(records)
        labels = [p.label for p in pairs]
        for i in range(0, len(labels), 2):
            self.assertEqual(labels[i], "stable")
            self.assertEqual(labels[i + 1], "progressed")

    def test_image_id_matches_record(self):
        from dcc.data.image_pair_builder import build_image_pairs
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records(2, Path(tmpdir))
            pairs = build_image_pairs(records)
        self.assertEqual(pairs[0].image_id, records[0].image_id)
        self.assertEqual(pairs[1].image_id, records[0].image_id)
        self.assertEqual(pairs[2].image_id, records[1].image_id)

    def test_images_are_uint8(self):
        from dcc.data.image_pair_builder import build_image_pairs
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records(1, Path(tmpdir))
            pairs = build_image_pairs(records)
        pair = pairs[0]
        self.assertEqual(pair.baseline_image.dtype, np.uint8)
        self.assertEqual(pair.followup_image.dtype, np.uint8)

    def test_baseline_and_followup_images_differ(self):
        """Different perturb seeds should produce different images."""
        from dcc.data.image_pair_builder import build_image_pairs
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records(1, Path(tmpdir))
            pairs = build_image_pairs(records, image_perturb_seed=0)
        pair = pairs[0]
        self.assertFalse(
            (pair.baseline_image == pair.followup_image).all(),
            "Baseline and followup images should differ after different perturb configs.",
        )

    def test_annotations_are_dicts(self):
        from dcc.data.image_pair_builder import build_image_pairs
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records(1, Path(tmpdir))
            pairs = build_image_pairs(records)
        for pair in pairs:
            self.assertIsInstance(pair.baseline_annotation, dict)
            self.assertIsInstance(pair.followup_annotation, dict)
            self.assertIn("teeth", pair.baseline_annotation)
            self.assertIn("teeth", pair.followup_annotation)

    def test_empty_records_returns_empty_list(self):
        from dcc.data.image_pair_builder import build_image_pairs
        result = build_image_pairs([])
        self.assertEqual(result, [])

    def test_record_without_teeth_is_skipped(self):
        from dcc.data.image_pair_builder import build_image_pairs

        @dataclass
        class _EmptyRec:
            image_id: str = "empty"
            image_path: Path = Path("/dev/null")
            annotation_dict: dict = None

            def __post_init__(self):
                self.annotation_dict = {"teeth": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            good = _make_records(1, Path(tmpdir))
            records = [_EmptyRec()] + good
            pairs = build_image_pairs(records)
        # Only 1 good record → 2 pairs
        self.assertEqual(len(pairs), 2)

    def test_deterministic_given_same_seed(self):
        from dcc.data.image_pair_builder import build_image_pairs
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records(2, Path(tmpdir))
            pairs1 = build_image_pairs(records, image_perturb_seed=42)
            pairs2 = build_image_pairs(records, image_perturb_seed=42)
        for p1, p2 in zip(pairs1, pairs2):
            self.assertTrue((p1.baseline_image == p2.baseline_image).all())

    def test_different_perturb_seeds_produce_different_images(self):
        from dcc.data.image_pair_builder import build_image_pairs
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records(1, Path(tmpdir))
            pairs_a = build_image_pairs(records, image_perturb_seed=0)
            pairs_b = build_image_pairs(records, image_perturb_seed=999)
        same = (pairs_a[0].baseline_image == pairs_b[0].baseline_image).all()
        self.assertFalse(same)


class TestBuildImagePairsEarlyBreak(unittest.TestCase):
    """Cover lines 103 and 122 — early break when ann_pairs runs out."""

    def _fake_pair(self, label: str = "stable"):
        from dcc.perturb.acquisition import PerturbedPair
        ann = _make_annotation("1")
        return PerturbedPair(baseline=ann, followup=ann, label=label, params=None, true_change=0.0)

    def test_line_122_break_when_only_one_pair_available(self):
        """Line 122: after consuming the stable pair pair_index >= len(ann_pairs) → break."""
        from unittest.mock import patch
        from dcc.data.image_pair_builder import build_image_pairs

        fake_pairs = [self._fake_pair("stable")]  # only 1 pair for 1 record
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records(1, Path(tmpdir))
            with patch("dcc.data.image_pair_builder.build_pairs", return_value=fake_pairs):
                result = build_image_pairs(records)
        # stable pair consumed, line 122 triggers, progressed pair skipped
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "stable")

    def test_line_103_break_when_extra_records_exceed_pairs(self):
        """Line 103: second record finds pair_index >= len(ann_pairs) → break at top of loop."""
        from unittest.mock import patch
        from dcc.data.image_pair_builder import build_image_pairs

        # 2 pairs → first record consumes both; second record hits line 103
        fake_pairs = [self._fake_pair("stable"), self._fake_pair("progressed")]
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_records(2, Path(tmpdir))
            with patch("dcc.data.image_pair_builder.build_pairs", return_value=fake_pairs):
                result = build_image_pairs(records)
        # first record consumed both pairs; second record exited at line 103
        self.assertEqual(len(result), 2)


class TestImagePairBuilderHardDeps(unittest.TestCase):
    """numpy and PIL are hard dependencies: blocking either raises ImportError
    at module import time (no graceful fallback)."""

    _MOD_KEY = "dcc.data.image_pair_builder"

    def _reload_blocking(self, *names: str):
        """Reload image_pair_builder with the named modules blocked.

        Pre-imports transitive deps so only the target module is re-executed,
        then asserts ImportError propagates from the blocked import.
        """
        import importlib
        import sys
        from unittest.mock import patch

        for dep in (
            "dcc.data.pair_builder",
            "dcc.perturb.image_perturb",
            "dcc.perturb.acquisition",
            "dcc.geometry",
        ):
            importlib.import_module(dep)

        block = {name: None for name in names}

        saved = sys.modules.pop(self._MOD_KEY, None)
        self.addCleanup(self._restore_module, saved)

        with patch.dict(sys.modules, block):
            with self.assertRaises(ImportError):
                importlib.import_module(self._MOD_KEY)

    def _restore_module(self, saved):
        import sys
        sys.modules.pop(self._MOD_KEY, None)
        if saved is not None:
            sys.modules[self._MOD_KEY] = saved

    def test_missing_numpy_raises_import_error(self):
        self._reload_blocking("numpy")

    def test_missing_pil_raises_import_error(self):
        self._reload_blocking("PIL", "PIL.Image")


if __name__ == "__main__":
    unittest.main()
