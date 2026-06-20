#!/usr/bin/env python3
"""Generate a failure-case gallery from Gate 2 evaluation outputs.

Reads an outputs/ directory from run_gate2.py or run_gate2_denpar.py and
identifies the worst failure cases:
  - False progressions: stable pairs certified as progressed
  - Over-uncertain: progressed pairs left uncertain (missed detections)
  - Interval calibration failures: pairs where GT score falls outside interval

Writes a Markdown gallery to outputs/failure_gallery/ with per-case summaries.

Usage:
    python scripts/failure_gallery.py --input outputs/gate2_denpar
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_eval_rows(input_dir: Path) -> list[dict]:
    """Load evaluation rows from metrics.json if rows are embedded, else return []."""
    metrics_path = input_dir / "metrics.json"
    if not metrics_path.exists():
        return []
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    return data.get("rows", [])


def find_false_progressions(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("true") == "stable" and r.get("decision") == "progressed"]


def find_missed_detections(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("true") == "progressed" and r.get("decision") == "uncertain"]


def find_interval_failures(rows: list[dict]) -> list[dict]:
    """Rows where the GT (oracle) score falls outside the conformal interval.

    The oracle score is taken from an explicit ``true_change`` field when one
    is present; otherwise it comes from the ``gt_score`` produced by the
    store-based evaluation path (falling back to ``score``).  An interval
    "failure" is a real coverage failure: the true change lies outside the
    predicted ``[lo, hi]`` interval.
    """
    failures = []
    for r in rows:
        lo, hi = r.get("lo"), r.get("hi")
        if lo is None or hi is None:
            continue
        if "true_change" in r:
            gt = r["true_change"]
        elif r.get("gt_score") is not None:
            gt = r["gt_score"]
        else:
            gt = r.get("score", 0.0)
        if gt is None:
            continue
        if not (lo <= gt <= hi):
            failures.append({**r, "_true_change": gt})
    return failures


def write_gallery(
    false_progs: list[dict],
    missed: list[dict],
    interval_fails: list[dict],
    output_dir: Path,
    input_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = output_dir / "failure_gallery.md"

    lines = [
        f"# Failure-Case Gallery",
        f"",
        f"Source: `{input_dir}`",
        f"",
        f"## Summary",
        f"",
        f"| Category | Count |",
        f"|----------|-------|",
        f"| False progressions (stable → certified progressed) | {len(false_progs)} |",
        f"| Missed detections (progressed → uncertain) | {len(missed)} |",
        f"| Interval calibration failures | {len(interval_fails)} |",
        f"",
    ]

    def _row_table(rows: list[dict], title: str, n: int = 10) -> list[str]:
        out = [f"## {title} (worst {min(n, len(rows))} of {len(rows)})", ""]
        if not rows:
            return out + ["_None._", ""]
        # Sort by score descending for false progs, ascending for missed
        def _fmt(x: object) -> str:
            return f"{x:.4f}" if isinstance(x, (int, float)) else "?"

        out += ["| Score | Lo | Hi | GT Score |", "|-------|----|----|----------|"]
        for r in rows[:n]:
            score = r.get("score", r.get("predicted_score"))
            gt = r.get("gt_score", r.get("_true_change"))
            out.append(
                f"| {_fmt(score)} "
                f"| {_fmt(r.get('lo'))} "
                f"| {_fmt(r.get('hi'))} "
                f"| {_fmt(gt)} |"
            )
        out.append("")
        return out

    lines += _row_table(
        sorted(false_progs, key=lambda r: r.get("score", 0.0), reverse=True),
        "False Progressions"
    )
    lines += _row_table(
        sorted(missed, key=lambda r: r.get("score", 0.0)),
        "Missed Detections (uncertain when should be progressed)"
    )
    lines += _row_table(interval_fails, "Interval Calibration Failures")

    gallery_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gallery_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Failure-case gallery from Gate 2 outputs")
    parser.add_argument("--input", default="outputs/gate2_denpar",
                        help="Gate 2 output directory containing metrics.json")
    parser.add_argument("--output", default=None,
                        help="Gallery output directory (default: <input>/failure_gallery)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "failure_gallery"

    rows = load_eval_rows(input_dir)
    if not rows:
        print(f"No rows found in {input_dir}/metrics.json. Run a gate script first.")
        print("The gallery can only be generated after a full evaluation run.")
        sys.exit(0)

    false_progs = find_false_progressions(rows)
    missed = find_missed_detections(rows)
    interval_fails = find_interval_failures(rows)

    gallery_path = write_gallery(false_progs, missed, interval_fails, output_dir, input_dir)
    print(f"Gallery written to {gallery_path}")
    print(f"  False progressions:         {len(false_progs)}")
    print(f"  Missed detections:          {len(missed)}")
    print(f"  Interval calibration fails: {len(interval_fails)}")


if __name__ == "__main__":
    main()
