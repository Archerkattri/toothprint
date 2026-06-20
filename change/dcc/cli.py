"""Command line entry points for lightweight DentalChangeCert runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from dcc.artifacts.manifest import BenchmarkArtifact, BenchmarkArtifactManifest, write_artifact_manifest
from dcc.data.manifest import build_default_manifest, write_manifest
from dcc.benchmark.pipeline import evaluate_pairs
from dcc.splits.freeze import build_deterministic_split, write_split
from dcc.certificate.conformal import ConformalInterval
from dcc.eval.metrics import summarize_decisions
from dcc.eval.report import write_report
from dcc.perturb.acquisition import TransformParams, apply_acquisition_perturbation
from dcc.perturb.truechange import inject_crestal_change


_PERIO_KPT_DEFAULT_ROOT = Path("data/perio-kpt/extracted/perio_KPT")
_PERIO_KPT_DEFAULT_OUTPUT = Path("outputs/perio_kpt_manifest.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dcc", description="DentalChangeCert utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("write-manifest", help="Write dataset manifest JSON")
    manifest_parser.add_argument("--output", type=Path, default=Path("outputs/dataset_manifest.json"))

    split_parser = subparsers.add_parser("freeze-splits", help="Write a deterministic split manifest")
    split_parser.add_argument("--case-id", action="append", dest="case_ids", required=True)
    split_parser.add_argument("--output", type=Path, default=Path("outputs/splits.json"))
    split_parser.add_argument("--seed", default="dental-change-cert-v0")

    scaffold_parser = subparsers.add_parser("write-scaffold", help="Write GPU-ready reproducibility scaffold files")
    scaffold_parser.add_argument("--output-dir", type=Path, default=Path("outputs/scaffold"))

    demo_parser = subparsers.add_parser("run-demo", help="Write a tiny deterministic demo report")
    demo_parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))

    perio_kpt_parser = subparsers.add_parser(
        "extract-perio-kpt",
        help="Extract perio-KPT records into a manifest JSON",
    )
    perio_kpt_parser.add_argument(
        "--root",
        type=Path,
        default=_PERIO_KPT_DEFAULT_ROOT,
        help="Path to the extracted perio_KPT directory",
    )
    perio_kpt_parser.add_argument(
        "--output",
        type=Path,
        default=_PERIO_KPT_DEFAULT_OUTPUT,
        help="Output manifest JSON path",
    )
    perio_kpt_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only process first N images (for fast validation)",
    )

    args = parser.parse_args(argv)
    if args.command == "write-manifest":
        print(write_manifest(build_default_manifest(), args.output))
        return 0
    if args.command == "freeze-splits":
        print(write_split(build_deterministic_split(args.case_ids, seed=args.seed), args.output))
        return 0
    if args.command == "write-scaffold":
        manifest_path = write_manifest(build_default_manifest(), args.output_dir / "dataset_manifest.json")
        split_path = write_split(
            build_deterministic_split([f"fixture_{i:03d}" for i in range(12)]),
            args.output_dir / "splits.json",
        )
        artifact_path = write_artifact_manifest(
            BenchmarkArtifactManifest(
                schema_version="0.1",
                artifacts=[
                    BenchmarkArtifact(
                        id="dataset_manifest",
                        path=str(manifest_path),
                        kind="json",
                        description="Dataset source/license manifest.",
                        generated_by="dcc write-scaffold",
                    ),
                    BenchmarkArtifact(
                        id="frozen_splits",
                        path=str(split_path),
                        kind="json",
                        description="Deterministic train/calibration/test split scaffold.",
                        generated_by="dcc write-scaffold",
                    ),
                ],
            ),
            args.output_dir / "artifact_manifest.json",
        )
        print(manifest_path)
        print(split_path)
        print(artifact_path)
        return 0
    if args.command == "run-demo":
        return _run_demo(args.output_dir)
    if args.command == "extract-perio-kpt":
        return _run_extract_perio_kpt(root=args.root, output=args.output, limit=args.limit)
    raise ValueError(f"Unhandled command: {args.command}")  # pragma: no cover


def _run_demo(output_dir: Path) -> int:
    """Run the demo using real perio-KPT baseline records (first 3).

    Falls back to the hardcoded fixture annotation when the dataset is not
    present so that CI and users without the data can still exercise the
    pipeline.
    """
    annotations = _load_demo_annotations(limit=3)

    pairs = []
    for ann in annotations:
        tooth_id = _first_tooth_id(ann)
        pairs.append(apply_acquisition_perturbation(ann, TransformParams(dy=2.0)))
        pairs.append(inject_crestal_change(ann, tooth_id=tooth_id, delta_px=5.0))

    tau = 1.0
    rows = evaluate_pairs(pairs, tau=tau, conformal=ConformalInterval(radius=0.5, alpha=0.1))
    summary = summarize_decisions(rows)
    report_path, metrics_path = write_report(summary, output_dir)
    print(report_path)
    print(metrics_path)
    return 0


def _load_demo_annotations(limit: int = 3) -> list[dict]:
    """Return up to *limit* real perio-KPT baseline annotations, or fixture."""
    perio_root = _PERIO_KPT_DEFAULT_ROOT
    if perio_root.exists():
        from dcc.data.perio_kpt_adapter import PerioKptAdapter
        adapter = PerioKptAdapter(perio_root)
        annotations = []
        for record in adapter.records(split="baseline"):
            if len(annotations) >= limit:
                break
            ann = record.annotation_dict
            if ann.get("teeth"):
                annotations.append(ann)
        if annotations:
            return annotations
    # Dataset not available — fall back to fixture
    return [_demo_annotation()]


def _first_tooth_id(annotation: dict) -> str:
    """Return the tooth_id of the first tooth, or '36' as fallback."""
    teeth = annotation.get("teeth", [])
    if teeth:
        return str(teeth[0].get("tooth_id", "36"))
    return "36"


def _run_extract_perio_kpt(root: Path, output: Path, limit: int | None) -> int:
    """Implementation of the extract-perio-kpt CLI command."""
    import json
    import sys
    from dcc.data.perio_kpt_adapter import PerioKptAdapter
    from dcc.score.periodontal import record_change_scores

    if not root.exists():
        print(
            f"WARNING: perio-KPT root not found at {root}. "
            "Download and extract the dataset first.",
            file=sys.stderr,
        )
        return 1

    adapter = PerioKptAdapter(root)
    entries: list[dict] = []
    n_images = 0
    n_teeth = 0
    n_complete = 0

    for record in adapter.records():
        if limit is not None and n_images >= limit:
            break

        ann = record.annotation_dict
        try:
            record_change_scores(ann, ann)
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING: record_change_scores failed for {record.image_id}: {exc}",
                file=sys.stderr,
            )

        teeth = ann.get("teeth", [])
        has_cej = any(bool(t.get("cej")) for t in teeth)
        has_crest = any(bool(t.get("crest_line")) for t in teeth)
        has_apex = any(bool(t.get("apex")) for t in teeth)
        tooth_complete = sum(
            1 for t in teeth
            if bool(t.get("cej")) and bool(t.get("crest_line")) and bool(t.get("apex"))
        )

        entries.append({
            "image_id": record.image_id,
            "image_path": str(record.image_path),
            "annotation_path": str(getattr(record, "label_path", None) or ""),
            "split": record.split,
            "tooth_count": len(teeth),
            "has_cej": has_cej,
            "has_crest": has_crest,
            "has_apex": has_apex,
        })
        n_images += 1
        n_teeth += len(teeth)
        n_complete += tooth_complete

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"Processed: {n_images} images")
    print(f"Total teeth: {n_teeth}")
    print(f"Teeth with complete CEJ+crest+apex: {n_complete}")
    print(f"Manifest written to: {output}")
    return 0


def _demo_annotation() -> dict:
    return {
        "image": "case001.png",
        "teeth": [
            {
                "tooth_id": "36",
                "cej": [[10.0, 20.0], [30.0, 20.0]],
                "apex": [[20.0, 80.0]],
                "crest_line": [[10.0, 35.0], [30.0, 35.0]],
            }
        ],
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
