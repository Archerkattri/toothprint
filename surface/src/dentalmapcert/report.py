"""Markdown and JSONL report writers."""

from __future__ import annotations

import json
from pathlib import Path

from dentalmapcert.certificate import CertificateOutput


_SYNTHETIC_NOTE = (
    "> **Note:** Coverage scores in this report are derived from synthetic heuristics\n"
    "> (n_views × 0.18 base). Real reconstruction-based scores require running the\n"
    "> reconstruction pipeline (dentalmapcert.reconstruction.reconstruct_point_cloud)\n"
    "> on actual phone captures with a GPU; the demo CLI uses synthetic coverage only."
)


def render_report(
    certificates: list[CertificateOutput],
    synthetic: bool = True,
) -> str:
    counts: dict[str, int] = {}
    for cert in certificates:
        counts[cert.label] = counts.get(cert.label, 0) + 1
    lines = [
        "# DentalMapCert Gate Report",
        "",
    ]
    if synthetic:
        lines += [_SYNTHETIC_NOTE, ""]
    lines += [
        "## Label Counts",
        "",
    ]
    for label, n in sorted(counts.items()):
        lines.append(f"- {label}: {n}")
    lines.extend(
        [
            "",
            "## Certificates",
            "",
            "| surface | label | coverage t0 | coverage t1 | delta interval mm | recapture |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for cert in certificates:
        lines.append(
            "| "
            + " | ".join(
                [
                    cert.surface_region_id,
                    cert.label,
                    f"{cert.coverage_score_t0:.2f}",
                    f"{cert.coverage_score_t1:.2f}",
                    f"[{cert.delta_interval_mm[0]:.2f}, {cert.delta_interval_mm[1]:.2f}]",
                    ", ".join(cert.recapture_actions) or "-",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    certificates: list[CertificateOutput],
    output_dir: Path | str,
    synthetic: bool = True,
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.md"
    jsonl_path = out / "certificate_output.jsonl"
    report_path.write_text(render_report(certificates, synthetic=synthetic), encoding="utf-8")
    jsonl_path.write_text(
        "\n".join(json.dumps(cert.to_dict(), sort_keys=True) for cert in certificates) + "\n",
        encoding="utf-8",
    )
    return report_path, jsonl_path

