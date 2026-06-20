"""Tests for RealDenparAdapter and build_denpar_teeth / load_real_denpar."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Real DenPAR root – tests that need it are skipped when absent.
# Repo-relative by default (matches load_real_denpar's path convention);
# overridable via DCC_DENPAR_ROOT. No hardcoded personal absolute path.
# ---------------------------------------------------------------------------

_REAL_ROOT = Path(
    os.environ.get("DCC_DENPAR_ROOT", "data/denpar/extracted/Dataset")
)


def _real_root_available() -> bool:
    return _REAL_ROOT.is_dir()


# ---------------------------------------------------------------------------
# Integration tests against the actual DenPAR data
# ---------------------------------------------------------------------------


class TestLoadRealDenpar(unittest.TestCase):
    """Tests that require the real DenPAR dataset on disk."""

    def setUp(self) -> None:
        if not _real_root_available():
            self.skipTest(f"Real DenPAR root not found: {_REAL_ROOT}")

    def _all_records(self):
        from dcc.data.denpar_adapter import load_real_denpar

        return load_real_denpar(_REAL_ROOT)

    def test_returns_more_than_500_records(self) -> None:
        records = self._all_records()
        self.assertGreater(
            len(records),
            500,
            f"Expected >500 records, got {len(records)}",
        )

    def test_annotation_dicts_have_nonempty_teeth(self) -> None:
        from dcc.data.denpar_adapter import load_real_denpar

        records = load_real_denpar(_REAL_ROOT)
        empty_count = sum(
            1
            for r in records
            if not r.annotation_dict.get("teeth")
        )
        # Allow up to 5 % of images to have no matchable teeth (edge cases).
        self.assertLess(
            empty_count / len(records),
            0.05,
            f"{empty_count}/{len(records)} records have empty teeth list",
        )

    def test_each_tooth_has_required_keys(self) -> None:
        from dcc.data.denpar_adapter import load_real_denpar

        records = load_real_denpar(_REAL_ROOT)
        for rec in records:
            for tooth in rec.annotation_dict.get("teeth", []):
                self.assertIn(
                    "tooth_id",
                    tooth,
                    f"tooth_id missing in {rec.image_id}: {tooth}",
                )
                has_cej = "cej" in tooth and tooth["cej"]
                has_crest = "crest_line" in tooth and tooth["crest_line"]
                self.assertTrue(
                    has_cej or has_crest,
                    f"Tooth {tooth.get('tooth_id')} in {rec.image_id} "
                    "has neither cej nor crest_line",
                )

    def test_all_splits_are_valid(self) -> None:
        from dcc.data.denpar_adapter import load_real_denpar

        records = load_real_denpar(_REAL_ROOT)
        valid_splits = {"train", "val", "test"}
        for rec in records:
            self.assertIn(
                rec.split,
                valid_splits,
                f"Unexpected split {rec.split!r} for {rec.image_id}",
            )

    def test_split_filter_train_only(self) -> None:
        from dcc.data.denpar_adapter import load_real_denpar

        train_records = load_real_denpar(_REAL_ROOT, split="train")
        self.assertTrue(
            len(train_records) > 0,
            "Expected at least one train record",
        )
        for rec in train_records:
            self.assertEqual(
                rec.split,
                "train",
                f"Non-train record returned: {rec.image_id} split={rec.split}",
            )

    def test_split_filter_val_only(self) -> None:
        from dcc.data.denpar_adapter import load_real_denpar

        val_records = load_real_denpar(_REAL_ROOT, split="val")
        self.assertTrue(len(val_records) > 0, "Expected at least one val record")
        for rec in val_records:
            self.assertEqual(rec.split, "val")

    def test_split_filter_test_only(self) -> None:
        from dcc.data.denpar_adapter import load_real_denpar

        test_records = load_real_denpar(_REAL_ROOT, split="test")
        self.assertTrue(len(test_records) > 0, "Expected at least one test record")
        for rec in test_records:
            self.assertEqual(rec.split, "test")

    def test_split_counts_sum_to_total(self) -> None:
        from dcc.data.denpar_adapter import load_real_denpar

        all_records = load_real_denpar(_REAL_ROOT)
        train = load_real_denpar(_REAL_ROOT, split="train")
        val = load_real_denpar(_REAL_ROOT, split="val")
        test = load_real_denpar(_REAL_ROOT, split="test")
        self.assertEqual(
            len(all_records),
            len(train) + len(val) + len(test),
            "Per-split counts do not sum to total",
        )


# ---------------------------------------------------------------------------
# Unit tests with synthetic fixture data (no real data required)
# ---------------------------------------------------------------------------


class TestBuildAnnotationDictSynthetic(unittest.TestCase):
    """Unit tests for the spatial matching logic using hand-crafted data."""

    def _build(self, kp_data: dict, bl_data: dict) -> dict:
        from dcc.data.denpar_adapter import _build_annotation_dict

        return _build_annotation_dict(kp_data, bl_data)

    def _make_fixture(self) -> tuple[dict, dict]:
        """Two bboxes side-by-side.

        Layout (all coordinates are pixel values):

        Bbox 0: x=[100,300], y=[100,500]   centre=(200,300)
        Bbox 1: x=[350,550], y=[100,500]   centre=(450,300)

        CEJ points (4):
          (150,200), (250,200)  → both near bbox0 centre x=200
          (400,200), (500,200)  → both near bbox1 centre x=450

        Bone lines (2):
          line0: [(180,350),(220,360)]  centroid≈(200,355) → bbox0
          line1: [(430,350),(470,360)]  centroid≈(450,355) → bbox1

        Apex points (2):
          (200,480) → inside bbox0
          (450,480) → inside bbox1
        """
        kp_data = {
            "Image_id": "synth.jpg",
            "bboxes": [
                [100.0, 100.0, 300.0, 500.0],
                [350.0, 100.0, 550.0, 500.0],
            ],
            "CEJ_Points": [
                [150.0, 200.0],
                [250.0, 200.0],
                [400.0, 200.0],
                [500.0, 200.0],
            ],
            "Apex_Points": [
                [200.0, 480.0],
                [450.0, 480.0],
            ],
        }
        bl_data = {
            "Image_id": "synth.jpg",
            "Num_of_Bone_Lines": 2,
            "Bone_Lines": [
                [[180.0, 350.0], [220.0, 360.0]],
                [[430.0, 350.0], [470.0, 360.0]],
            ],
        }
        return kp_data, bl_data

    def test_produces_two_teeth(self) -> None:
        kp_data, bl_data = self._make_fixture()
        result = self._build(kp_data, bl_data)
        self.assertEqual(len(result["teeth"]), 2)

    def test_image_id_propagated(self) -> None:
        kp_data, bl_data = self._make_fixture()
        result = self._build(kp_data, bl_data)
        self.assertEqual(result["image"], "synth.jpg")

    def test_tooth_ids_are_1_indexed_strings(self) -> None:
        kp_data, bl_data = self._make_fixture()
        result = self._build(kp_data, bl_data)
        tooth_ids = [t["tooth_id"] for t in result["teeth"]]
        self.assertEqual(tooth_ids, ["1", "2"])

    def test_crest_line_endpoints_for_tooth_0(self) -> None:
        """Tooth 0 should get crest_line = [[180,350],[220,360]]."""
        kp_data, bl_data = self._make_fixture()
        result = self._build(kp_data, bl_data)
        tooth0 = result["teeth"][0]
        self.assertEqual(tooth0["crest_line"], [[180.0, 350.0], [220.0, 360.0]])

    def test_crest_line_endpoints_for_tooth_1(self) -> None:
        """Tooth 1 should get crest_line = [[430,350],[470,360]]."""
        kp_data, bl_data = self._make_fixture()
        result = self._build(kp_data, bl_data)
        tooth1 = result["teeth"][1]
        self.assertEqual(tooth1["crest_line"], [[430.0, 350.0], [470.0, 360.0]])

    def test_cej_for_tooth_0_has_two_points_sorted_by_x(self) -> None:
        kp_data, bl_data = self._make_fixture()
        result = self._build(kp_data, bl_data)
        tooth0_cej = result["teeth"][0]["cej"]
        self.assertEqual(len(tooth0_cej), 2)
        self.assertLessEqual(tooth0_cej[0][0], tooth0_cej[1][0])
        # The two points nearest bbox0 x-centre (200) from its window [40,360]:
        # all four points are in the window, pick 2 closest to 200 → (150,200) & (250,200)
        xs = sorted(p[0] for p in tooth0_cej)
        self.assertEqual(xs, [150.0, 250.0])

    def test_apex_present_for_both_teeth(self) -> None:
        kp_data, bl_data = self._make_fixture()
        result = self._build(kp_data, bl_data)
        for tooth in result["teeth"]:
            self.assertIn("apex", tooth)
            self.assertEqual(len(tooth["apex"]), 1)

    def test_apex_inside_bbox_tooth_0(self) -> None:
        kp_data, bl_data = self._make_fixture()
        result = self._build(kp_data, bl_data)
        apex = result["teeth"][0]["apex"][0]
        self.assertEqual(apex, [200.0, 480.0])

    def test_apex_inside_bbox_tooth_1(self) -> None:
        kp_data, bl_data = self._make_fixture()
        result = self._build(kp_data, bl_data)
        apex = result["teeth"][1]["apex"][0]
        self.assertEqual(apex, [450.0, 480.0])

    def test_skip_tooth_when_both_cej_and_crest_line_empty(self) -> None:
        """A bbox with no nearby CEJ and no bone line should be skipped."""
        kp_data = {
            "Image_id": "skip.jpg",
            # Bbox far from all annotations
            "bboxes": [[5000.0, 5000.0, 5100.0, 5100.0]],
            "CEJ_Points": [[100.0, 100.0]],
            "Apex_Points": [],
        }
        bl_data = {
            "Image_id": "skip.jpg",
            "Num_of_Bone_Lines": 0,
            "Bone_Lines": [],
        }
        result = self._build(kp_data, bl_data)
        # No bone lines → crest_line empty; CEJ point x=100 is outside
        # [5000-60, 5100+60] = [4940, 5160] → cej empty too → tooth skipped.
        self.assertEqual(result["teeth"], [])

    def test_no_apex_key_when_no_apex_inside_bbox(self) -> None:
        """When no apex falls inside a bbox the 'apex' key must be omitted."""
        kp_data = {
            "Image_id": "noapex.jpg",
            "bboxes": [[100.0, 100.0, 300.0, 300.0]],
            "CEJ_Points": [[150.0, 150.0], [250.0, 150.0]],
            "Apex_Points": [[200.0, 9999.0]],  # y outside bbox
        }
        bl_data = {
            "Image_id": "noapex.jpg",
            "Num_of_Bone_Lines": 1,
            "Bone_Lines": [[[120.0, 200.0], [280.0, 200.0]]],
        }
        result = self._build(kp_data, bl_data)
        self.assertEqual(len(result["teeth"]), 1)
        self.assertNotIn("apex", result["teeth"][0])


class TestRealDenparRecordDataclass(unittest.TestCase):
    def test_frozen_dataclass_fields(self) -> None:
        from dcc.data.denpar_adapter import RealDenparRecord

        rec = RealDenparRecord(
            image_id="0001",
            image_path=Path("/tmp/0001.jpg"),
            kp_path=Path("/tmp/0001_kp.json"),
            bl_path=Path("/tmp/0001_bl.json"),
            annotation_dict={"image": "0001.jpg", "teeth": []},
            split="train",
        )
        self.assertEqual(rec.image_id, "0001")
        self.assertEqual(rec.split, "train")

    def test_frozen_dataclass_is_immutable(self) -> None:
        from dcc.data.denpar_adapter import RealDenparRecord

        rec = RealDenparRecord(
            image_id="0001",
            image_path=Path("/tmp/0001.jpg"),
            kp_path=Path("/tmp/0001_kp.json"),
            bl_path=Path("/tmp/0001_bl.json"),
            annotation_dict={},
            split="val",
        )
        with self.assertRaises(Exception):
            rec.split = "train"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
