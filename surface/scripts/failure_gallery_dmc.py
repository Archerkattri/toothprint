#!/usr/bin/env python3
"""Generate a failure-case gallery from DMC Gate 1 evaluation outputs.

Reads the JSONL report from run_dmc_gate1_real.py and identifies:
  - Over-uncertain: regions certified as "uncertain / recapture" (too wide interval)
  - False-change: stable regions certified as "surface change certified"

Usage:
    python scripts/failure_gallery_dmc.py --input outputs/dmc_gate1_real
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/dmc_gate1_real")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "failure_gallery"

    # Find report JSONL
    jsonl_files = list(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {input_dir}. Run run_dmc_gate1_real.py first.")
        sys.exit(0)

    records = []
    for jsonl_path in jsonl_files:
        for line in jsonl_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not records:
        print(f"No records found in {input_dir}/*.jsonl")
        sys.exit(0)

    over_uncertain = [r for r in records if r.get("label") == "uncertain / recapture"]
    false_change = [r for r in records if r.get("label") == "surface change certified"]
    stable_certified = [r for r in records if r.get("label") == "surface stable certified"]
    not_claimable = [r for r in records if r.get("label") == "not visible / not claimable"]
    _known = {
        "uncertain / recapture", "surface change certified",
        "surface stable certified", "not visible / not claimable",
    }
    other = [r for r in records if r.get("label") not in _known]

    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = output_dir / "failure_gallery_dmc.md"

    # Detect synthetic provenance from the sibling report.md the gate writes
    # (it embeds a "synthetic" note when run without real captures), so the
    # gallery never presents generated rows as real failure analysis.
    report_md = input_dir / "report.md"
    is_synthetic = (
        report_md.exists() and "synthetic" in report_md.read_text(encoding="utf-8").lower()
    )
    if is_synthetic:
        print(
            "[SYNTHETIC] gallery rows are generated (no real captures); not real "
            "failure analysis.",
            file=sys.stderr,
        )

    lines = [
        "# DMC Failure-Case Gallery",
        "",
        f"Source: `{input_dir}`",
        "",
    ]
    if is_synthetic:
        lines += [
            "> **SYNTHETIC DATA** — these failure cases are generated from the "
            "fallback pipeline (no real phone captures), not from real imagery.",
            "",
        ]
    lines += [
        "## Summary",
        "",
        "| Category | Count |",
        "|----------|-------|",
        f"| Surface stable certified | {len(stable_certified)} |",
        f"| Surface change certified (potential false-change) | {len(false_change)} |",
        f"| Uncertain / recapture (over-cautious) | {len(over_uncertain)} |",
        f"| Not visible / not claimable | {len(not_claimable)} |",
        f"| Other / unrecognised label | {len(other)} |",
        f"| Total records | {len(records)} |",
        "",
    ]

    def _fmt(x: object) -> str:
        return f"{x:.2f}" if isinstance(x, (int, float)) else "?"

    def _table(certs: list[dict], title: str, n: int = 10) -> list[str]:
        out = [f"## {title} (first {min(n, len(certs))} of {len(certs)})", ""]
        if not certs:
            return out + ["_None._", ""]
        out += ["| Region | Coverage T0 | Coverage T1 | Delta Interval | Recapture |",
                "|--------|------------|------------|----------------|-----------|"]
        for c in certs[:n]:
            di = c.get("delta_interval_mm")
            if not isinstance(di, (list, tuple)) or len(di) < 2:
                di = ["?", "?"]
            out.append(
                f"| {c.get('surface_region_id','?')} "
                f"| {_fmt(c.get('coverage_score_t0'))} "
                f"| {_fmt(c.get('coverage_score_t1'))} "
                f"| [{_fmt(di[0])}, {_fmt(di[1])}] "
                f"| {', '.join(c.get('recapture_actions', [])) or 'none'} |"
            )
        out.append("")
        return out

    lines += _table(false_change, "False-Change Certificates")
    lines += _table(
        sorted(over_uncertain, key=lambda r: r.get("coverage_score_t0", 0.0)),
        "Over-Uncertain Regions (lowest coverage first)"
    )

    gallery_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Gallery written to {gallery_path}")
    print(f"  Stable certified:    {len(stable_certified)}")
    print(f"  Change certified:    {len(false_change)}")
    print(f"  Uncertain/recapture: {len(over_uncertain)}")
    print(f"  Not claimable:       {len(not_claimable)}")
    if not (false_change or over_uncertain):
        print(
            f"WARNING: gallery has no false-change or over-uncertain rows "
            f"({len(records)} records: {len(stable_certified)} stable, "
            f"{len(not_claimable)} not-claimable, {len(other)} other). "
            f"Nothing actionable to display.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
