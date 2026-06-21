"""Markdown reports for DentalChangeCert gates."""

from __future__ import annotations

import json
from pathlib import Path

from toothprint.bench.eval.metrics import DECISIONS, LABELS, DecisionSummary, coverage_vs_false_progression_curve


def render_markdown_report(summary: DecisionSummary) -> str:
    lines = [
        "# DentalChangeCert Gate Report",
        "",
        "## Primary Metrics",
        "",
        f"- False progression rate: {summary.false_progression_rate:.3f}",
        f"- True change recall: {summary.true_change_recall:.3f}",
        f"- Uncertain rate: {summary.uncertain_rate:.3f}",
        f"- Mean interval width: {summary.mean_interval_width:.3f}",
        f"- Interval width std: {summary.interval_width_std:.3f}",
        "",
        "## 2x3 Outcome Table",
        "",
        "| true | stable | progressed | uncertain |",
        "|---|---:|---:|---:|",
    ]
    for label in LABELS:
        cells = [str(summary.table[label][decision]) for decision in DECISIONS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_report(
    summary: DecisionSummary,
    output_dir: Path | str,
    rows: list[dict] | None = None,
    tau: float | None = None,
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.md"
    metrics_path = out / "metrics.json"
    report_path.write_text(render_markdown_report(summary), encoding="utf-8")

    metrics_dict = summary.to_dict()
    if rows is not None and tau is not None:
        metrics_dict["fpr_curve"] = coverage_vs_false_progression_curve(rows, tau=tau)

    if rows is not None:
        # Serialize rows for failure gallery consumption
        serializable_rows = []
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                serializable_rows.append({k: v for k, v in row.__dict__.items() if v is not None})
            else:
                serializable_rows.append(dict(row))
        metrics_dict["rows"] = serializable_rows

    metrics_path.write_text(json.dumps(metrics_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path, metrics_path
