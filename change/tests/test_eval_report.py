import json
import tempfile
import unittest
from pathlib import Path


class EvalReportTests(unittest.TestCase):
    def test_metrics_build_2_by_3_outcome_table(self):
        from dcc.eval.metrics import summarize_decisions

        summary = summarize_decisions(
            [
                {"true": "stable", "decision": "stable"},
                {"true": "stable", "decision": "progressed"},
                {"true": "progressed", "decision": "progressed"},
                {"true": "progressed", "decision": "uncertain"},
            ]
        )

        self.assertEqual(summary.table["stable"]["stable"], 1)
        self.assertEqual(summary.table["stable"]["progressed"], 1)
        self.assertEqual(summary.table["progressed"]["progressed"], 1)
        self.assertEqual(summary.false_progression_rate, 0.5)
        self.assertEqual(summary.true_change_recall, 0.5)
        self.assertEqual(summary.uncertain_rate, 0.25)

    def test_report_contains_primary_metrics(self):
        from dcc.eval.metrics import summarize_decisions
        from dcc.eval.report import render_markdown_report

        summary = summarize_decisions(
            [
                {"true": "stable", "decision": "stable"},
                {"true": "progressed", "decision": "progressed"},
            ]
        )
        report = render_markdown_report(summary)

        self.assertIn("# DentalChangeCert Gate Report", report)
        self.assertIn("False progression rate", report)
        self.assertIn("| stable |", report)

    def test_report_serializes_rows(self):
        from dcc.eval.metrics import summarize_decisions
        from dcc.eval.report import write_report

        rows = [
            {"true": "stable", "decision": "stable", "score": 1.0, "lo": 0.0, "hi": 2.0},
            {"true": "progressed", "decision": "progressed", "score": 8.0, "lo": 6.0, "hi": 10.0},
        ]
        summary = summarize_decisions(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            _, metrics_path = write_report(summary, out_dir, rows=rows, tau=10.0)
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertIn("rows", data)
            self.assertEqual(len(data["rows"]), 2)
            self.assertEqual(data["rows"][0]["true"], "stable")
            self.assertEqual(data["rows"][1]["true"], "progressed")


    def test_write_report_with_dataclass_row_serializes_fields(self):
        """write_report serializes dataclass rows via __dict__ (eval/report.py line 56)."""
        from dcc.benchmark.pipeline import EvalRow
        from dcc.eval.metrics import summarize_decisions
        from dcc.eval.report import write_report

        row = EvalRow(
            true="stable", decision="stable", score=1.0, lo=0.0, hi=2.0,
            predicted_score=1.0, gt_score=None,
        )
        summary_rows = [{"true": "stable", "decision": "stable", "score": 1.0, "lo": 0.0, "hi": 2.0}]
        summary = summarize_decisions(summary_rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            _, metrics_path = write_report(summary, Path(tmpdir), rows=[row])
            data = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertIn("rows", data)
        self.assertEqual(data["rows"][0]["true"], "stable")


if __name__ == "__main__":
    unittest.main()
