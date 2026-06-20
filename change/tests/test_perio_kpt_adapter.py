"""Tests for the perio-KPT YOLO-keypoint adapter and image perturbations."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers to build synthetic test fixtures
# ---------------------------------------------------------------------------

def _make_png(path: Path, width: int = 200, height: int = 150) -> Path:
    """Write a minimal valid PNG with the given dimensions."""
    import zlib

    def _u32be(n: int) -> bytes:
        return struct.pack(">I", n)

    def _chunk(name: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return _u32be(len(data)) + name + data + _u32be(crc)

    ihdr_data = (
        _u32be(width)
        + _u32be(height)
        + bytes([8, 2, 0, 0, 0])  # bit depth=8, color=RGB, compression=0, filter=0, interlace=0
    )

    # Minimal IDAT: one row of black pixels
    raw_row = bytes([0]) + bytes([0] * width * 3)  # filter byte + RGB pixels
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


def _make_yolo_line(
    cls: int = 0,
    kp1: tuple[float, float, float] = (0.1, 0.2, 2.0),  # CEJ left
    kp2: tuple[float, float, float] = (0.3, 0.2, 2.0),  # CEJ right
    kp3: tuple[float, float, float] = (0.0, 0.0, 0.0),
    kp4: tuple[float, float, float] = (0.1, 0.4, 2.0),  # crest mesial
    kp5: tuple[float, float, float] = (0.3, 0.4, 2.0),  # crest distal
    kp6: tuple[float, float, float] = (0.0, 0.0, 0.0),
    kp7: tuple[float, float, float] = (0.2, 0.9, 2.0),  # apex
    kp8: tuple[float, float, float] = (0.0, 0.0, 0.0),
    kp9: tuple[float, float, float] = (0.0, 0.0, 0.0),
    kp10: tuple[float, float, float] = (0.0, 0.0, 0.0),
    kp11: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cx: float = 0.2, cy: float = 0.55, bw: float = 0.25, bh: float = 0.75,
) -> str:
    parts = [str(cls), str(cx), str(cy), str(bw), str(bh)]
    for kp in (kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10, kp11):
        parts.extend([str(v) for v in kp])
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Unit tests — YOLO parsing (no file I/O required)
# ---------------------------------------------------------------------------

class YoloParsingTests(unittest.TestCase):
    """Test _parse_yolo_line with a synthetic 38-value line."""

    def _parse(self, line: str, width: int = 200, height: int = 150) -> dict | None:
        from dcc.data.perio_kpt_adapter import _parse_yolo_line
        return _parse_yolo_line(line, width=width, height=height, index=0)

    def test_cej_coords_converted_to_pixels(self):
        # kp1 = (0.1, 0.2, 2.0) → pixel (0.1*200, 0.2*150) = (20.0, 30.0)
        # kp2 = (0.3, 0.2, 2.0) → pixel (0.3*200, 0.2*150) = (60.0, 30.0)
        line = _make_yolo_line()
        tooth = self._parse(line, width=200, height=150)
        self.assertIsNotNone(tooth)
        cej = tooth["cej"]
        self.assertEqual(len(cej), 2)
        self.assertAlmostEqual(cej[0][0], 20.0, places=4)
        self.assertAlmostEqual(cej[0][1], 30.0, places=4)
        self.assertAlmostEqual(cej[1][0], 60.0, places=4)
        self.assertAlmostEqual(cej[1][1], 30.0, places=4)

    def test_crest_line_coords_converted_to_pixels(self):
        # kp4 = (0.1, 0.4, 2.0) → pixel (20.0, 60.0)
        # kp5 = (0.3, 0.4, 2.0) → pixel (60.0, 60.0)
        line = _make_yolo_line()
        tooth = self._parse(line, width=200, height=150)
        crest = tooth["crest_line"]
        self.assertEqual(len(crest), 2)
        self.assertAlmostEqual(crest[0][0], 20.0, places=4)
        self.assertAlmostEqual(crest[0][1], 60.0, places=4)

    def test_apex_coords_converted_to_pixels(self):
        # kp7 = (0.2, 0.9, 2.0) → pixel (40.0, 135.0)
        line = _make_yolo_line()
        tooth = self._parse(line, width=200, height=150)
        self.assertIn("apex", tooth)
        apex = tooth["apex"]
        self.assertEqual(len(apex), 1)
        self.assertAlmostEqual(apex[0][0], 40.0, places=4)
        self.assertAlmostEqual(apex[0][1], 135.0, places=4)

    def test_tooth_id_is_one_based(self):
        from dcc.data.perio_kpt_adapter import _parse_yolo_line
        line = _make_yolo_line()
        tooth = _parse_yolo_line(line, width=200, height=150, index=0)
        self.assertEqual(tooth["tooth_id"], "1")
        tooth2 = _parse_yolo_line(line, width=200, height=150, index=4)
        self.assertEqual(tooth2["tooth_id"], "5")

    def test_invisible_keypoints_excluded(self):
        # Set kp1 visibility=0 (not labeled) → CEJ should have only one point
        line = _make_yolo_line(kp1=(0.1, 0.2, 0.0))
        tooth = self._parse(line, width=200, height=150)
        self.assertEqual(len(tooth["cej"]), 1)

    def test_teeth_with_no_cej_and_no_crest_are_skipped(self):
        # All CEJ and crest points invisible → tooth is None
        line = _make_yolo_line(
            kp1=(0.0, 0.0, 0.0),
            kp2=(0.0, 0.0, 0.0),
            kp4=(0.0, 0.0, 0.0),
            kp5=(0.0, 0.0, 0.0),
        )
        tooth = self._parse(line)
        self.assertIsNone(tooth)

    def test_tooth_without_apex_has_no_apex_key(self):
        line = _make_yolo_line(kp7=(0.0, 0.0, 0.0))
        tooth = self._parse(line)
        self.assertNotIn("apex", tooth)

    def test_short_line_returns_none(self):
        tooth = self._parse("0 0.1 0.2 0.3 0.4")
        self.assertIsNone(tooth)


# ---------------------------------------------------------------------------
# Integration tests against real extracted data
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    Path("data/perio-kpt/extracted/perio_KPT").exists(),
    "perio-KPT data not available at data/perio-kpt/extracted/perio_KPT/",
)
class PerioKptAdapterRealDataTests(unittest.TestCase):
    _ROOT = Path("data/perio-kpt/extracted/perio_KPT")

    def _adapter(self):
        from dcc.data.perio_kpt_adapter import PerioKptAdapter
        return PerioKptAdapter(self._ROOT)

    def test_baseline_split_yields_records(self):
        records = list(self._adapter().records(split="baseline"))
        self.assertGreater(len(records), 0, "Expected at least one baseline record")

    def test_all_records_have_non_empty_teeth(self):
        adapter = self._adapter()
        # Check up to 20 records for speed
        records = []
        for i, r in enumerate(adapter.records()):
            records.append(r)
            if i >= 19:
                break
        for record in records:
            self.assertGreater(
                len(record.annotation_dict.get("teeth", [])),
                0,
                f"Expected teeth for {record.image_id}",
            )

    def test_record_split_values_are_valid(self):
        valid_splits = {"baseline", "experiment", "holdout", "external"}
        adapter = self._adapter()
        for i, record in enumerate(adapter.records()):
            self.assertIn(record.split, valid_splits)
            if i >= 49:
                break

    def test_cej_coordinates_are_pixel_scale(self):
        """Pixel coords for a ~200×150+ image should be > 1.0 when present."""
        adapter = self._adapter()
        for record in adapter.records(split="baseline"):
            for tooth in record.annotation_dict.get("teeth", []):
                for pt in tooth.get("cej", []):
                    self.assertGreater(
                        max(abs(pt[0]), abs(pt[1])),
                        1.0,
                        f"CEJ coordinates look normalized (not pixel) for {record.image_id}",
                    )
            break  # one image is enough

    def test_record_change_scores_runs_without_error(self):
        from dcc.score.periodontal import record_change_scores
        adapter = self._adapter()
        for i, record in enumerate(adapter.records()):
            ann = record.annotation_dict
            # Use identical baseline/followup → all changes should be 0
            scores = record_change_scores(ann, ann)
            for tooth_id, score in scores.items():
                self.assertAlmostEqual(
                    score, 0.0, places=6,
                    msg=f"Self-comparison should yield zero change for tooth {tooth_id}",
                )
            if i >= 9:
                break

    def test_split_filter_returns_correct_subset(self):
        adapter = self._adapter()
        baseline_records = list(adapter.records(split="baseline"))
        for r in baseline_records[:5]:
            self.assertEqual(r.split, "baseline")


# ---------------------------------------------------------------------------
# Adapter unit test using a synthetic file layout
# ---------------------------------------------------------------------------

class PerioKptAdapterSyntheticTests(unittest.TestCase):
    def _build_fixture(self, tmpdir: Path) -> Path:
        root = tmpdir / "perio_KPT"
        img_dir = root / "0_Baseline" / "images"
        lbl_dir = root / "0_Baseline" / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        _make_png(img_dir / "TestImage1.png", width=200, height=150)
        label_line = _make_yolo_line()
        (lbl_dir / "TestImage1.txt").write_text(label_line + "\n", encoding="utf-8")
        return root

    def test_adapter_yields_one_record_for_synthetic_baseline(self):
        from dcc.data.perio_kpt_adapter import PerioKptAdapter
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._build_fixture(Path(tmpdir))
            adapter = PerioKptAdapter(root)
            records = list(adapter.records(split="baseline"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].image_id, "TestImage1")
        self.assertEqual(records[0].split, "baseline")

    def test_adapter_annotation_dict_structure(self):
        from dcc.data.perio_kpt_adapter import PerioKptAdapter
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._build_fixture(Path(tmpdir))
            adapter = PerioKptAdapter(root)
            records = list(adapter.records())
        ann = records[0].annotation_dict
        self.assertEqual(ann["image"], "TestImage1.png")
        self.assertIn("teeth", ann)
        tooth = ann["teeth"][0]
        self.assertIn("cej", tooth)
        self.assertIn("crest_line", tooth)
        self.assertIn("apex", tooth)

    def test_record_change_scores_on_synthetic_annotation(self):
        from dcc.data.perio_kpt_adapter import PerioKptAdapter
        from dcc.score.periodontal import record_change_scores
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._build_fixture(Path(tmpdir))
            adapter = PerioKptAdapter(root)
            records = list(adapter.records())
        ann = records[0].annotation_dict
        scores = record_change_scores(ann, ann)
        for score in scores.values():
            self.assertAlmostEqual(score, 0.0, places=6)

    def test_missing_root_yields_no_records(self):
        from dcc.data.perio_kpt_adapter import PerioKptAdapter
        adapter = PerioKptAdapter(Path("/nonexistent/path"))
        records = list(adapter.records())
        self.assertEqual(records, [])


# ---------------------------------------------------------------------------
# Image perturbation tests
# ---------------------------------------------------------------------------

class ImagePerturbConfigTests(unittest.TestCase):
    def test_random_config_is_deterministic_from_seed(self):
        from dcc.perturb.image_perturb import random_image_perturb_config
        cfg1 = random_image_perturb_config(42)
        cfg2 = random_image_perturb_config(42)
        self.assertEqual(cfg1, cfg2)

    def test_different_seeds_produce_different_configs(self):
        from dcc.perturb.image_perturb import random_image_perturb_config
        cfg1 = random_image_perturb_config(1)
        cfg2 = random_image_perturb_config(2)
        # Extremely unlikely to be equal
        self.assertNotEqual(cfg1, cfg2)

    def test_brightness_delta_in_range(self):
        from dcc.perturb.image_perturb import random_image_perturb_config
        for seed in range(20):
            cfg = random_image_perturb_config(seed)
            self.assertGreaterEqual(cfg.brightness_delta, -0.3)
            self.assertLessEqual(cfg.brightness_delta, 0.3)

    def test_contrast_scale_in_range(self):
        from dcc.perturb.image_perturb import random_image_perturb_config
        for seed in range(20):
            cfg = random_image_perturb_config(seed)
            self.assertGreaterEqual(cfg.contrast_scale, 0.8)
            self.assertLessEqual(cfg.contrast_scale, 1.2)

    def test_noise_std_in_range(self):
        from dcc.perturb.image_perturb import random_image_perturb_config
        for seed in range(20):
            cfg = random_image_perturb_config(seed)
            self.assertGreaterEqual(cfg.noise_std, 0.0)
            self.assertLessEqual(cfg.noise_std, 0.05)


class ImagePerturbApplicationTests(unittest.TestCase):
    """Test that apply_image_perturbation keeps values in [0, 1] and is stable."""

    def _make_image(self, h: int = 4, w: int = 6, c: int = 3):
        """Return a small HxWxC float array using numpy if available."""
        try:
            import numpy as np
            rng = np.random.default_rng(0)
            return rng.random((h, w, c))
        except ImportError:
            import random
            rng = random.Random(0)
            return [[[rng.random() for _ in range(c)] for _ in range(w)] for _ in range(h)]

    def _min_max(self, arr):
        try:
            import numpy as np
            if isinstance(arr, np.ndarray):
                return float(arr.min()), float(arr.max())
        except ImportError:
            pass
        values = [v for row in arr for pixel in row for v in pixel]
        return min(values), max(values)

    def test_output_values_are_in_0_1_after_brightness_perturbation(self):
        from dcc.perturb.image_perturb import ImagePerturbConfig, apply_image_perturbation
        cfg = ImagePerturbConfig(brightness_delta=0.3)
        img = self._make_image()
        out = apply_image_perturbation(img, cfg)
        lo, hi = self._min_max(out)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_output_values_are_in_0_1_after_noise_perturbation(self):
        from dcc.perturb.image_perturb import ImagePerturbConfig, apply_image_perturbation
        cfg = ImagePerturbConfig(noise_std=0.05)
        img = self._make_image()
        out = apply_image_perturbation(img, cfg)
        lo, hi = self._min_max(out)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_output_values_are_in_0_1_after_combined_perturbation(self):
        from dcc.perturb.image_perturb import random_image_perturb_config, apply_image_perturbation
        cfg = random_image_perturb_config(99)
        img = self._make_image()
        out = apply_image_perturbation(img, cfg)
        lo, hi = self._min_max(out)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_identity_config_preserves_values(self):
        from dcc.perturb.image_perturb import ImagePerturbConfig, apply_image_perturbation
        cfg = ImagePerturbConfig()  # all defaults: no-op
        try:
            import numpy as np
            img = np.array([[[0.5, 0.5, 0.5]]], dtype=float)
            out = apply_image_perturbation(img, cfg)
            self.assertAlmostEqual(float(out[0, 0, 0]), 0.5, places=6)
        except ImportError:
            img = [[[0.5, 0.5, 0.5]]]
            out = apply_image_perturbation(img, cfg)
            self.assertAlmostEqual(out[0][0][0], 0.5, places=6)

    def test_flip_horizontal_reverses_columns(self):
        from dcc.perturb.image_perturb import ImagePerturbConfig, apply_image_perturbation
        cfg = ImagePerturbConfig(flip_horizontal=True)
        try:
            import numpy as np
            # 1x3x1 image: col values [0.1, 0.5, 0.9]
            img = np.array([[[0.1], [0.5], [0.9]]], dtype=float)
            out = apply_image_perturbation(img, cfg)
            self.assertAlmostEqual(float(out[0, 0, 0]), 0.9, places=6)
            self.assertAlmostEqual(float(out[0, 2, 0]), 0.1, places=6)
        except ImportError:
            img = [[[0.1], [0.5], [0.9]]]
            out = apply_image_perturbation(img, cfg)
            self.assertAlmostEqual(out[0][0][0], 0.9, places=6)
            self.assertAlmostEqual(out[0][2][0], 0.1, places=6)

    def test_perturbation_output_shape_matches_input(self):
        from dcc.perturb.image_perturb import random_image_perturb_config, apply_image_perturbation
        cfg = random_image_perturb_config(7)
        h, w, c = 8, 10, 3
        try:
            import numpy as np
            img = np.zeros((h, w, c))
            out = apply_image_perturbation(img, cfg)
            self.assertEqual(out.shape, (h, w, c))
        except ImportError:
            img = [[[0.0] * c for _ in range(w)] for _ in range(h)]
            out = apply_image_perturbation(img, cfg)
            self.assertEqual(len(out), h)
            self.assertEqual(len(out[0]), w)
            self.assertEqual(len(out[0][0]), c)


# ---------------------------------------------------------------------------
# PNG/JPEG size reader tests
# ---------------------------------------------------------------------------

class ImageSizeReaderTests(unittest.TestCase):
    def test_reads_png_dimensions(self):
        from dcc.data.perio_kpt_adapter import _read_image_size
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _make_png(img_path, width=320, height=240)
            w, h = _read_image_size(img_path)
        self.assertEqual(w, 320)
        self.assertEqual(h, 240)

    def test_returns_zeros_for_nonexistent_file(self):
        from dcc.data.perio_kpt_adapter import _read_image_size
        w, h = _read_image_size(Path("/nonexistent/image.png"))
        self.assertEqual((w, h), (0, 0))


class PerioKptAdapterExperimentIterEdgeCases(unittest.TestCase):
    """Cover missing lines 106, 111 in PerioKptAdapter._iter_experiment."""

    def test_non_dir_and_non_f_items_in_standard_box_are_skipped(self):
        """Line 106: files and dirs not starting with 'f' inside standard_box are skipped."""
        from dcc.data.perio_kpt_adapter import PerioKptAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "perio_KPT"
            std_box = root / "1_Experiment" / "standard_box"
            std_box.mkdir(parents=True)
            # A plain file — not is_dir() → skip
            (std_box / "readme.txt").write_text("not a fold")
            # A dir not starting with "f" → skip
            (std_box / "other_dir").mkdir()

            adapter = PerioKptAdapter(root)
            records = list(adapter.records(split="experiment"))
        self.assertEqual(records, [])

    def test_fold_dir_without_images_subdir_is_skipped(self):
        """Line 111: fold dir with no train/images subdir → images_dir.is_dir() False → continue."""
        from dcc.data.perio_kpt_adapter import PerioKptAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "perio_KPT"
            fold_dir = root / "1_Experiment" / "standard_box" / "f0"
            # create train dir but no images/ inside it
            (fold_dir / "train").mkdir(parents=True)

            adapter = PerioKptAdapter(root)
            records = list(adapter.records(split="experiment"))
        self.assertEqual(records, [])


class PerioKptImageLabelDirEdgeCases(unittest.TestCase):
    """Cover missing lines 139, 142, 146, 150, 177."""

    def _make_baseline_dir(self, root: Path) -> tuple:
        img_dir = root / "0_Baseline" / "images"
        lbl_dir = root / "0_Baseline" / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        return img_dir, lbl_dir

    def test_non_image_suffix_skipped(self):
        """Line 139: .txt files in images dir are not yielded."""
        from dcc.data.perio_kpt_adapter import PerioKptAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "perio_KPT"
            img_dir, lbl_dir = self._make_baseline_dir(root)
            (img_dir / "document.txt").write_text("not an image")

            adapter = PerioKptAdapter(root)
            records = list(adapter.records(split="baseline"))
        self.assertEqual(records, [])

    def test_image_without_label_is_skipped(self):
        """Line 142: image with no matching .txt label is skipped."""
        from dcc.data.perio_kpt_adapter import PerioKptAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "perio_KPT"
            img_dir, lbl_dir = self._make_baseline_dir(root)
            _make_png(img_dir / "img001.png", width=200, height=150)
            # deliberately no img001.txt in lbl_dir

            adapter = PerioKptAdapter(root)
            records = list(adapter.records(split="baseline"))
        self.assertEqual(records, [])

    def test_unreadable_image_size_is_skipped(self):
        """Line 146: image that returns (0,0) size is skipped."""
        from dcc.data.perio_kpt_adapter import PerioKptAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "perio_KPT"
            img_dir, lbl_dir = self._make_baseline_dir(root)
            # Write a .png file with invalid PNG signature
            (img_dir / "bad.png").write_bytes(b"\x00" * 20)
            # Write a matching label so it passes line 142
            (lbl_dir / "bad.txt").write_text(_make_yolo_line() + "\n")

            adapter = PerioKptAdapter(root)
            records = list(adapter.records(split="baseline"))
        self.assertEqual(records, [])

    def test_label_file_producing_no_teeth_is_skipped(self):
        """Line 150: label file with no valid teeth lines yields no record."""
        from dcc.data.perio_kpt_adapter import PerioKptAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "perio_KPT"
            img_dir, lbl_dir = self._make_baseline_dir(root)
            _make_png(img_dir / "img001.png", width=200, height=150)
            # Label with only blank lines → _parse_yolo_label returns []
            (lbl_dir / "img001.txt").write_text("\n\n\n", encoding="utf-8")

            adapter = PerioKptAdapter(root)
            records = list(adapter.records(split="baseline"))
        self.assertEqual(records, [])

    def test_empty_lines_in_label_are_skipped(self):
        """Line 177: blank lines inside a label file are silently skipped."""
        from dcc.data.perio_kpt_adapter import _parse_yolo_label

        with tempfile.TemporaryDirectory() as tmpdir:
            lbl = Path(tmpdir) / "label.txt"
            # blank line + valid line + blank line
            lbl.write_text("\n" + _make_yolo_line() + "\n\n", encoding="utf-8")
            teeth = _parse_yolo_label(lbl, width=200, height=150)
        self.assertEqual(len(teeth), 1)


class PNGSizeEdgeCases(unittest.TestCase):
    """Cover line 257: _png_size returns (0,0) for bad PNG signature."""

    def test_bad_png_signature_returns_zeros(self):
        """Line 257: file with invalid PNG signature → return 0, 0."""
        from dcc.data.perio_kpt_adapter import _png_size

        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.png"
            bad.write_bytes(b"\x00" * 20)
            result = _png_size(bad)
        self.assertEqual(result, (0, 0))


class JPEGSizeEdgeCases(unittest.TestCase):
    """Cover lines 273, 277, 284, 290, 293 in _jpeg_size."""

    def test_bad_soi_marker_returns_zeros(self):
        """Line 273: file not starting with 0xFF 0xD8 → return 0, 0."""
        from dcc.data.perio_kpt_adapter import _jpeg_size

        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.jpg"
            bad.write_bytes(b"\x00\xd8\xff\xe0")
            result = _jpeg_size(bad)
        self.assertEqual(result, (0, 0))

    def test_non_0xff_byte_in_stream_causes_break(self):
        """Lines 277, 293: non-0xFF byte at position i → break → return 0, 0."""
        from dcc.data.perio_kpt_adapter import _jpeg_size

        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.jpg"
            # SOI (\xff\xd8) + garbage byte (\x00) + extra byte
            bad.write_bytes(b"\xff\xd8\x00\xff")
            result = _jpeg_size(bad)
        self.assertEqual(result, (0, 0))

    def test_truncated_sof_marker_causes_break(self):
        """Line 284: SOF marker present but not enough bytes for h/w → break → 0, 0."""
        from dcc.data.perio_kpt_adapter import _jpeg_size

        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.jpg"
            # SOI + SOF0 marker + only 5 more bytes (need 6 from current i)
            bad.write_bytes(b"\xff\xd8\xff\xc0" + b"\x00" * 5)
            result = _jpeg_size(bad)
        self.assertEqual(result, (0, 0))

    def test_truncated_segment_causes_break(self):
        """Lines 290, 293: non-SOF segment with no room for length bytes → break → 0, 0."""
        from dcc.data.perio_kpt_adapter import _jpeg_size

        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.jpg"
            # SOI + APP0 marker, no segment length bytes follow
            bad.write_bytes(b"\xff\xd8\xff\xe0")
            result = _jpeg_size(bad)
        self.assertEqual(result, (0, 0))

    def test_no_sof_found_returns_zeros(self):
        """Line 293: valid JPEG structure but no SOF marker → while exits → 0, 0."""
        from dcc.data.perio_kpt_adapter import _jpeg_size
        import struct

        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.jpg"
            # SOI + APP0 with length=4 (consuming 4 bytes), then data ends
            # After parsing APP0: i = 4 + 4 = 8 > len(data)-1 → loop exits
            seg_len = struct.pack(">H", 4)
            bad.write_bytes(b"\xff\xd8\xff\xe0" + seg_len)
            result = _jpeg_size(bad)
        self.assertEqual(result, (0, 0))


if __name__ == "__main__":
    unittest.main()
