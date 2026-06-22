"""Dataset source manifest for the DentalChangeCert open benchmark."""

from __future__ import annotations

import json
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class DatasetSource:
    id: str
    name: str
    url: str
    doi: str | None
    license: str
    modality: str
    annotation_types: list[str]
    access: str
    role: str
    redistribution_allowed: bool
    limitations: str


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    sources: list[DatasetSource]

    def get_source(self, source_id: str) -> DatasetSource:
        for source in self.sources:
            if source.id == source_id:
                return source
        raise KeyError(f"Unknown dataset source: {source_id}")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "sources": [asdict(source) for source in self.sources],
        }


def build_default_manifest() -> DatasetManifest:
    """Return the project source manifest pinned by the current Gate 0 plan."""
    return DatasetManifest(
        schema_version=SCHEMA_VERSION,
        sources=[
            DatasetSource(
                id="denpar",
                name="DenPAR",
                url="https://www.nature.com/articles/s41597-025-05906-9",
                doi="10.5281/zenodo.16645076",
                license="CC-BY-4.0",
                modality="Intraoral periapical radiographs",
                annotation_types=[
                    "tooth masks",
                    "CEJ keypoints",
                    "apex keypoints",
                    "alveolar crest lines",
                    "metadata",
                ],
                access="public",
                role="core",
                redistribution_allowed=True,
                limitations=(
                    "Single-timepoint public data; supports controlled synthetic-pair "
                    "stress tests, not clinical longitudinal outcome claims."
                ),
            ),
            DatasetSource(
                id="mendeley_bitewing_caries",
                name="Bitewing Radiography Dental Caries Dataset",
                url="https://data.mendeley.com/datasets/4fbdxs7s7w/1",
                doi="10.17632/4fbdxs7s7w.1",
                license="CC-BY-NC-3.0",
                modality="Bitewing radiographs",
                annotation_types=["caries bounding boxes", "8 annotator labels"],
                access="public-noncommercial",
                role="optional-nc",
                redistribution_allowed=False,
                limitations=(
                    "Excluded from open release because the license is non-commercial; "
                    "only valid for a quarantined optional annotator-disagreement study."
                ),
            ),
            DatasetSource(
                id="periapical_lesions_radiographs",
                name="Dataset of dental radiographs for the study of periapical lesions",
                url="https://zenodo.org/records/13772918",
                doi="10.5281/zenodo.13772918",
                license="CC-BY-4.0",
                modality="Conventional diagnostic periapical radiographs of anterior teeth",
                annotation_types=[
                    "periapical lesion binary labels",
                    "5 specialist classifications",
                ],
                access="public",
                role="external-shift",
                redistribution_allowed=True,
                limitations=(
                    "Single-timepoint anterior-tooth periapical data digitized with "
                    "multiple cameras; useful for periapical domain shift and lesion "
                    "confounding, not clinical longitudinal bone-level certificate claims."
                ),
            ),
        ],
    )


def write_manifest(manifest: DatasetManifest, output_path: Path) -> Path:
    """Write a manifest JSON file and return the path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def source_ids(sources: Iterable[DatasetSource]) -> set[str]:
    return {source.id for source in sources}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the DentalChangeCert dataset manifest."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dataset_manifest.json"),
        help="Path to write dataset_manifest.json",
    )
    args = parser.parse_args(argv)
    output_path = write_manifest(build_default_manifest(), args.output)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
