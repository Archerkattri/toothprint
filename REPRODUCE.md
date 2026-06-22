# Reproducing the evaluations

Every benchmark dataset is large, license-gated, and **gitignored**, so the result JSONs in
`evaluation/results/` are produced from data that is not in this repo. This document makes the
pipeline reproducible anyway: configurable data paths, committed synthetic fixtures for an
end-to-end smoke run, and one-command verification — no "trust the JSON" required to confirm the
code path works.

## 1. Smoke test — no data needed

The 3D identity pipeline runs end-to-end on the tiny synthetic arches committed under
`evaluation/fixtures/` (regenerate with `make_fixtures.py`):

```bash
TOOTHPRINT_FIXTURES=1 PYTHONPATH=. python evaluation/scripts/smoke_test.py
#  -> fixture identity: Rank-1 1.000  ...  SMOKE OK
```

This exercises `align_rigid` + the surface-distance identity path with **zero off-machine data**.
The fixtures are toy parametric geometry (per-subject tooth-bump signatures), not the benchmark —
they prove the code runs and separates genuine from impostor, nothing more.

## 2. Point the scripts at your data

Each dataset directory is resolved in `evaluation/scripts/paths.py`, overridable by env var
(defaults match the maintainer's layout — set the var to run elsewhere with no source edits):

| env var | dataset | layout |
|---|---|---|
| `TP_POSEIDON3D` | Poseidon3D intraoral STL arches | `<id>/<id>*.stl` |
| `TP_TEETH3DS` | Teeth3DS+ intraoral OBJ arches | `<id>/<id>*.obj` |
| `TP_CBCT_IOS` | paired CBCT (`.nii.gz`) + IOS (`.stl`) | `<id>/<id>_cbct/…`, `<id>/<id>_ios/…` |
| `TP_DENPAR` | DenPAR periapical radiographs + masks | `{Training,Testing,Validation}/Images/*.jpg` |

```bash
TP_POSEIDON3D=/data/poseidon3d PYTHONPATH=. python evaluation/scripts/eval_id3d.py
```

Dataset sources (all open or request-gated) are listed in `evaluation/EXTERNAL_DATA.md`.

## 3. Reproduce the headline result (#1 — learned partial-overlap correspondence)

Needs a GPU and the Poseidon3D arches:

```bash
TP_POSEIDON3D=/data/poseidon3d PYTHONPATH=. python evaluation/scripts/train_correspondence.py
TP_POSEIDON3D=/data/poseidon3d PYTHONPATH=. python evaluation/scripts/eval_correspondence.py
#  -> correspondence_identity.json: teeth-dropout keep-0.5 ~0.87, keep-0.3 ~0.57
```

Reference baselines printed alongside (crop-hardened, rigid GICP) are read at runtime from the
committed `embedding_partial.json` / `id3d.json`, not pasted constants.

## Honest scope

These reproduce the **simulation** results — synthetic re-scans and synthetic partial crops, since
the public datasets are single-timepoint. The one thing no code closes is real
cross-session/longitudinal data (#7); see "Help wanted — real longitudinal data" in the README.
The cross-dataset test (`eval_correspondence_teeth3ds.py`) also shows the learned descriptors carry
a real domain gap (in-domain 0.87 → cross-dataset 0.42), so the headline is in-domain.
