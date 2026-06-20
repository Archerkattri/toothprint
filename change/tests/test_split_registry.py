"""Tests for dcc/data/split_registry.py — 6 tests."""
from __future__ import annotations
import unittest
from dataclasses import dataclass


@dataclass
class _FakeRecord:
    image_id: str
    value: int = 0


class TestSplitRegistry(unittest.TestCase):

    def test_deterministic(self):
        """Same image_id + salt must always return the same split."""
        from dcc.data.split_registry import assign_split

        result_a = assign_split("case_001")
        result_b = assign_split("case_001")
        self.assertEqual(result_a, result_b)

    def test_different_salts_differ(self):
        """Different salts should produce different assignments for at least some ids."""
        from dcc.data.split_registry import assign_split

        # Use many ids to make it extremely unlikely all are the same
        ids = [f"img_{i:04d}" for i in range(50)]
        splits_v1 = [assign_split(iid, salt="dcc_v1") for iid in ids]
        splits_v2 = [assign_split(iid, salt="dcc_v2") for iid in ids]
        # At least some assignments should differ
        self.assertNotEqual(splits_v1, splits_v2)

    def test_fractions_approximately_correct(self):
        """1000 deterministic ids → ~70% train, ~15% val, ~15% test (±5%)."""
        from dcc.data.split_registry import assign_split, Split

        ids = [f"sample_{i:06d}" for i in range(1000)]
        counts = {Split.TRAIN: 0, Split.VAL: 0, Split.TEST: 0}
        for iid in ids:
            counts[assign_split(iid)] += 1

        n = len(ids)
        self.assertAlmostEqual(counts[Split.TRAIN] / n, 0.70, delta=0.05)
        self.assertAlmostEqual(counts[Split.VAL] / n, 0.15, delta=0.05)
        self.assertAlmostEqual(counts[Split.TEST] / n, 0.15, delta=0.05)

    def test_split_records_returns_all(self):
        """No records must be lost — total across splits equals input length."""
        from dcc.data.split_registry import split_records

        records = [_FakeRecord(image_id=f"rec_{i:04d}", value=i) for i in range(200)]
        result = split_records(records)
        total = sum(len(v) for v in result.values())
        self.assertEqual(total, len(records))

    def test_split_records_with_dict(self):
        """split_records should also work when records are plain dicts."""
        from dcc.data.split_registry import split_records

        records = [{"image_id": f"dict_{i:04d}", "extra": i} for i in range(100)]
        result = split_records(records)
        total = sum(len(v) for v in result.values())
        self.assertEqual(total, len(records))
        # All keys should be plain split strings
        self.assertSetEqual(set(result.keys()), {"train", "val", "test"})

    def test_split_enum_values(self):
        """Split enum string values must be 'train', 'val', 'test'."""
        from dcc.data.split_registry import Split

        self.assertEqual(Split.TRAIN.value, "train")
        self.assertEqual(Split.VAL.value, "val")
        self.assertEqual(Split.TEST.value, "test")


class FrozenSplitTests(unittest.TestCase):
    def _build_split(self):
        from dcc.splits.freeze import build_deterministic_split

        return build_deterministic_split(
            [f"case_{i:03d}" for i in range(20)],
            seed="test-seed",
            train_fraction=0.6,
            calibration_fraction=0.2,
        )

    def test_ids_for_returns_train_ids(self):
        """ids_for('train') returns non-empty list of case IDs."""
        fs = self._build_split()
        train_ids = fs.ids_for("train")
        self.assertIsInstance(train_ids, list)
        self.assertGreater(len(train_ids), 0)
        self.assertTrue(all(isinstance(i, str) for i in train_ids))

    def test_build_raises_on_bad_train_fraction(self):
        from dcc.splits.freeze import build_deterministic_split

        with self.assertRaises(ValueError, msg="train_fraction must be in (0, 1)"):
            build_deterministic_split(["a", "b", "c", "d"], train_fraction=0.0)
        with self.assertRaises(ValueError):
            build_deterministic_split(["a", "b", "c", "d"], train_fraction=1.0)

    def test_build_raises_on_bad_calibration_fraction(self):
        from dcc.splits.freeze import build_deterministic_split

        with self.assertRaises(ValueError, msg="calibration_fraction must be in (0, 1)"):
            build_deterministic_split(["a", "b", "c", "d"], calibration_fraction=0.0)

    def test_build_raises_when_fractions_leave_no_test(self):
        from dcc.splits.freeze import build_deterministic_split

        with self.assertRaises(ValueError, msg="train + calibration fractions must leave a test split"):
            build_deterministic_split(
                ["a", "b", "c", "d", "e"],
                train_fraction=0.6,
                calibration_fraction=0.5,
            )

    def test_build_raises_on_too_few_cases(self):
        from dcc.splits.freeze import build_deterministic_split

        with self.assertRaises(ValueError, msg="at least three cases"):
            build_deterministic_split(["only_one", "only_two"])


if __name__ == "__main__":
    unittest.main()
