"""Central, configurable data-path resolution for the eval scripts.

Every benchmark dataset is large, license-gated, and gitignored, so the scripts cannot hardcode
one machine's layout and still be reproducible elsewhere. Each dataset directory is therefore
resolved here, overridable by an environment variable, with the maintainer's layout as the
default:

    TP_POSEIDON3D   intraoral STL arches (Poseidon3D)        -> <id>/<id>*.stl
    TP_TEETH3DS     intraoral OBJ arches (Teeth3DS+)         -> <id>/<id>*.obj
    TP_CBCT_IOS     paired CBCT (.nii.gz) + IOS (.stl)
    TP_DENPAR       DenPAR periapical radiographs + masks

Point those at your copies and every eval runs unchanged. Or set TOOTHPRINT_FIXTURES=1 to use the
tiny synthetic fixtures committed under evaluation/fixtures/ — that runs the whole 3D identity /
correspondence pipeline end-to-end with NO off-machine data (used by smoke_test.py). The fixtures
exercise the code paths; they are not the benchmark and produce only toy numbers.
"""
from __future__ import annotations

import os
from pathlib import Path

_HOME = Path.home()
_FIX = Path(__file__).resolve().parents[1] / "fixtures"
USING_FIXTURES = bool(os.environ.get("TOOTHPRINT_FIXTURES"))


def _resolve(env: str, default: Path) -> Path:
    return Path(os.environ[env]) if env in os.environ else default


if USING_FIXTURES:
    POSEIDON3D = TEETH3DS = _FIX / "arches"
    CBCT_IOS = _FIX / "cbct_ios"
    DENPAR = _FIX / "denpar"
else:
    POSEIDON3D = _resolve("TP_POSEIDON3D", _HOME / "personal-projects/dental-map-cert/data/poseidon3d/extracted/data")
    TEETH3DS = _resolve("TP_TEETH3DS", _HOME / "personal-projects/toothprint-data/teeth3ds/extracted/upper")
    CBCT_IOS = _resolve("TP_CBCT_IOS", _HOME / "personal-projects/toothprint-data/cbct_ios_multimodal/extracted")
    DENPAR = _resolve("TP_DENPAR", _HOME / "personal-projects/dental-change-certificate/data/denpar/extracted/Dataset")
