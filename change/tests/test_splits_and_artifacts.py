import json
import tempfile
import unittest
from pathlib import Path

from dcc.artifacts.manifest import BenchmarkArtifact, BenchmarkArtifactManifest, write_artifact_manifest
from dcc.calibration.protocol import CalibrationBudget, CalibrationRecord, check_calibration_budget
from dcc.splits.freeze import build_deterministic_split, write_split


class SplitAndArtifactTests(unittest.TestCase):
    def test_split_is_deterministic_and_has_all_three_partitions(self):
        case_ids = [f"case_{i}" for i in range(20)]
        first = build_deterministic_split(case_ids, seed="fixed")
        second = build_deterministic_split(reversed(case_ids), seed="fixed")

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertGreater(first.counts()["train"], 0)
        self.assertGreater(first.counts()["calibration"], 0)
        self.assertGreater(first.counts()["test"], 0)

    def test_split_writer_outputs_counts(self):
        split = build_deterministic_split(["a", "b", "c", "d", "e"])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_split(split, Path(tmp) / "splits.json")
            payload = json.loads(path.read_text())

        self.assertEqual(payload["schema_version"], "0.1")
        self.assertIn("counts", payload)

    def test_calibration_budget_reports_small_strata(self):
        records = [
            CalibrationRecord("artifact", 1.0, 1.2, "fixture"),
            CalibrationRecord("artifact", 2.0, 2.1, "fixture"),
        ]
        report = check_calibration_budget(records=records, budget=CalibrationBudget(min_per_stratum=3))

        self.assertFalse(report["artifact"]["ok"])
        self.assertEqual(report["artifact"]["n"], 2)

    def test_check_calibration_budget_simple_form_sufficient(self):
        """FIX 6: n_cal=100, alpha=0.1 is sufficient for 0.9 coverage."""
        report = check_calibration_budget(n_cal=100, alpha=0.1, target_coverage=0.9)
        self.assertTrue(report.coverage_satisfiable)
        self.assertLessEqual(report.n_needed, 100)

    def test_check_calibration_budget_simple_form_insufficient(self):
        """FIX 6: n_cal=1 is insufficient for 0.9 coverage."""
        report = check_calibration_budget(n_cal=1, alpha=0.1, target_coverage=0.9)
        self.assertFalse(report.coverage_satisfiable)
        self.assertGreater(report.n_needed, 1)

    def test_check_calibration_budget_has_n_needed_attribute(self):
        """FIX 6: BudgetResult exposes .n_needed for the run_gate2.py print statement."""
        report = check_calibration_budget(n_cal=50, alpha=0.1)
        self.assertTrue(hasattr(report, "n_needed"))
        self.assertTrue(hasattr(report, "coverage_satisfiable"))

    def test_artifact_manifest_writes_gpu_contract(self):
        manifest = BenchmarkArtifactManifest(
            schema_version="0.1",
            artifacts=[
                BenchmarkArtifact(
                    id="demo",
                    path="outputs/demo/report.md",
                    kind="markdown",
                    description="Demo report",
                    generated_by="test",
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_artifact_manifest(manifest, Path(tmp) / "artifact_manifest.json")
            payload = json.loads(path.read_text())

        self.assertTrue(payload["gpu_required"])
        self.assertEqual(payload["artifacts"][0]["id"], "demo")


if __name__ == "__main__":
    unittest.main()
