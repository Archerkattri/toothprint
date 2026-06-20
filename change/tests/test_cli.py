import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    def test_demo_report_cli_writes_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            env = dict(os.environ)
            env.update({"CUDA_VISIBLE_DEVICES": "0"})

            subprocess.run(
                [sys.executable, "-m", "dcc.cli", "run-demo", "--output-dir", str(output_dir)],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            report = (output_dir / "report.md").read_text(encoding="utf-8")
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertIn("DentalChangeCert Gate Report", report)
        self.assertIn("false_progression_rate", metrics)


class CliDirectCallTests(unittest.TestCase):
    """Call main() directly (same process) so coverage tracks cli.py lines."""

    def test_write_manifest_subcommand(self):
        """main(['write-manifest', '--output', path]) → returns 0, writes JSON."""
        from dcc.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "manifest.json"
            result = main(["write-manifest", "--output", str(out)])
            self.assertEqual(result, 0)
            self.assertTrue(out.exists())
            payload = json.loads(out.read_text())
        self.assertEqual(payload["schema_version"], "0.1")

    def test_freeze_splits_subcommand(self):
        """main(['freeze-splits', ...]) → returns 0, writes split JSON."""
        from dcc.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "splits.json"
            result = main(["freeze-splits", "--case-id", "c001", "--case-id", "c002", "--case-id", "c003", "--output", str(out)])
            self.assertEqual(result, 0)
            self.assertTrue(out.exists())

    def test_write_scaffold_subcommand(self):
        """main(['write-scaffold', ...]) → returns 0, writes manifest+split+artifact JSON."""
        from dcc.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "scaffold"
            result = main(["write-scaffold", "--output-dir", str(out_dir)])
            self.assertEqual(result, 0)
            self.assertTrue((out_dir / "dataset_manifest.json").exists())
            self.assertTrue((out_dir / "splits.json").exists())
            self.assertTrue((out_dir / "artifact_manifest.json").exists())

    def test_run_demo_subcommand_writes_report(self):
        """main(['run-demo', ...]) → returns 0, writes report.md and metrics.json."""
        from dcc.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            result = main(["run-demo", "--output-dir", str(out_dir)])
            self.assertEqual(result, 0)
            self.assertTrue((out_dir / "report.md").exists())
            self.assertTrue((out_dir / "metrics.json").exists())

    def test_extract_perio_kpt_missing_root_returns_1(self):
        """main(['extract-perio-kpt', '--root', nonexistent]) → prints warning, returns 1."""
        from dcc.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            result = main([
                "extract-perio-kpt",
                "--root", "/nonexistent/perio_kpt_path_xyz",
                "--output", str(Path(tmpdir) / "out.json"),
            ])
        self.assertEqual(result, 1)


class CliExtractPerioKptExceptionPathTests(unittest.TestCase):
    """Cover lines 190-191 in _run_extract_perio_kpt (exception from record_change_scores)."""

    def test_record_change_scores_exception_is_warned_and_skipped(self):
        """Lines 190-191: if record_change_scores raises, print warning and continue."""
        from unittest.mock import patch
        from dcc.cli import main

        real_root = Path("data/perio-kpt/extracted/perio_KPT")
        if not real_root.exists():
            self.skipTest("perio-KPT data not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "extract.json"
            with patch("dcc.score.periodontal.record_change_scores", side_effect=RuntimeError("test-error")):
                result = main([
                    "extract-perio-kpt",
                    "--root", str(real_root),
                    "--output", str(out),
                    "--limit", "1",
                ])
            self.assertEqual(result, 0)


class CliHelperFunctionTests(unittest.TestCase):
    """Tests for private CLI helper functions."""

    def test_first_tooth_id_returns_first_tooth(self):
        """_first_tooth_id returns the tooth_id of the first tooth."""
        from dcc.cli import _first_tooth_id
        ann = {"teeth": [{"tooth_id": "36"}, {"tooth_id": "46"}]}
        self.assertEqual(_first_tooth_id(ann), "36")

    def test_first_tooth_id_fallback_when_no_teeth(self):
        """_first_tooth_id returns '36' fallback when teeth list is empty."""
        from dcc.cli import _first_tooth_id
        self.assertEqual(_first_tooth_id({"teeth": []}), "36")
        self.assertEqual(_first_tooth_id({}), "36")

    def test_demo_annotation_has_required_keys(self):
        """_demo_annotation() returns a valid annotation dict."""
        from dcc.cli import _demo_annotation
        ann = _demo_annotation()
        self.assertIn("image", ann)
        self.assertIn("teeth", ann)
        self.assertGreater(len(ann["teeth"]), 0)

    def test_load_demo_annotations_falls_back_to_fixture_when_root_absent(self):
        """Line 151: _load_demo_annotations returns fixture when perio-KPT root is absent."""
        from pathlib import Path
        from unittest.mock import patch
        from dcc.cli import _load_demo_annotations
        with patch("dcc.cli._PERIO_KPT_DEFAULT_ROOT", Path("/nonexistent/perio_kpt_xyz")):
            annotations = _load_demo_annotations(limit=3)
        self.assertIsInstance(annotations, list)
        self.assertGreater(len(annotations), 0)
        self.assertIn("teeth", annotations[0])

    def test_extract_perio_kpt_with_limit_covers_iteration_body(self):
        """Lines 177-225: _run_extract_perio_kpt processes records up to limit."""
        from dcc.cli import main
        real_root = Path("data/perio-kpt/extracted/perio_KPT")
        if not real_root.exists():
            self.skipTest("perio-KPT data not available")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "extract.json"
            result = main([
                "extract-perio-kpt",
                "--root", str(real_root),
                "--output", str(out),
                "--limit", "2",
            ])
            self.assertEqual(result, 0)
            payload = json.loads(out.read_text())
        self.assertIsInstance(payload, list)
        self.assertLessEqual(len(payload), 2)


if __name__ == "__main__":
    unittest.main()
