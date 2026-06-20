"""Paper-facing artifact manifest."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkArtifact:
    id: str
    path: str
    kind: str
    description: str
    generated_by: str


@dataclass(frozen=True)
class BenchmarkArtifactManifest:
    schema_version: str
    artifacts: list[BenchmarkArtifact]
    gpu_required: bool = True

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "gpu_required": self.gpu_required,
            "artifacts": [asdict(item) for item in self.artifacts],
        }


def write_artifact_manifest(manifest: BenchmarkArtifactManifest, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
