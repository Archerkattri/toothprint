# DATA — download, extract, reproduce

This file documents the **exact** download + extract steps that place each
dataset into the paths the scripts and loaders expect under the gitignored
`data/` directory, plus the exact commands that regenerate the
reconstruction-error and certificate results.

> Nothing here is run automatically. `scripts/fetch_data.sh` only downloads and
> extracts data; it does not run any GPU/reconstruction job. Datasets and
> weights are large — review the script before running it.

## Expected paths (what the code reads)

The loaders and scripts read these exact paths (all under `data/`):

| Dataset | Loader | Expected path | Contents |
|---|---|---|---|
| Poseidon3D | `Poseidon3DLoader` | `data/poseidon3d/extracted/data/` | `metadata.json` + per-case `*/` dirs with STL meshes |
| 3DTeethLand | `TeethLandLoader` | `data/teeth3ds/extracted/` | `upper/` and `lower/` dirs of per-case landmark JSON |
| Teeth3DS OBJ | `Teeth3DSLoader` | `data/teeth3ds/obj/` + `data/teeth3ds/labels/` | `*.obj` meshes + paired label files |
| Phone captures | `PhoneCaptureLoader` | `data/phone-captures/<subject>/<timepoint>/*.jpg` | project-owned smartphone photos |

- **Poseidon3D**: `metadata.json` lives at `extracted/data/metadata.json`; the
  STL relative paths inside it resolve against `extracted/` (the parent of
  `data/`). So the on-disk layout is
  `data/poseidon3d/extracted/data/<case_id>/<case_id>_MODEL_<arch>.stl`.
- **3DTeethLand**: each case is
  `data/teeth3ds/extracted/<upper|lower>/<case_id>/<case_id>_<arch>__kpt.json`
  (the loader also accepts any `*.json` in the case dir as a fallback).
- **Teeth3DS OBJ** meshes are gated behind Grand Challenge registration (see
  below); only the 3DTeethLand landmark JSONs are openly downloadable.

## Sources

| Dataset | Source | License / access |
|---|---|---|
| Poseidon3D | Zenodo record **15608906** | CC-BY-4.0 (open download) |
| 3DTeethLand | OSF project **um96h** (training split) | open download |
| Teeth3DS OBJ | <https://teeth3ds.grand-challenge.org/> | requires Grand Challenge registration |

> Confirm the current Zenodo/OSF archive filenames before running — record
> assets are occasionally re-versioned. `scripts/fetch_data.sh` reads the URLs
> from environment variables so you can override them without editing the
> script.

## Download + extract

```bash
# Optional: override archive URLs (defaults point at the records above).
# export POSEIDON3D_URL="https://zenodo.org/records/15608906/files/<archive>.zip"
# export TEETHLAND_URL="https://osf.io/<asset>/download"

bash scripts/fetch_data.sh             # downloads + extracts into data/*/extracted/
bash scripts/fetch_data.sh --poseidon  # only Poseidon3D
bash scripts/fetch_data.sh --teethland # only 3DTeethLand
```

After extraction, validate the paths without running any reconstruction:

```bash
python - <<'PY'
from dentalmapcert.dataset_loaders import Poseidon3DLoader, TeethLandLoader
for name, loader in [
    ("poseidon3d", Poseidon3DLoader("data/poseidon3d/extracted/data")),
    ("3dteethland", TeethLandLoader("data/teeth3ds/extracted")),
]:
    errs = loader.validate_paths()
    print(name, "OK" if not errs else f"{len(errs)} errors")
    for e in errs[:5]:
        print("  -", e)
PY
```

## Reproduce the results

All commands write to the gitignored `outputs/` directory.

### Reconstruction surface error (Poseidon3D)

```bash
# Metric default (VGGT) — needs a GPU (target: 2x RTX 5090, 32 GB each):
python scripts/run_dmc_reconstruction_real.py \
    --data data/poseidon3d/extracted/data \
    --output outputs/dmc_reconstruction_real \
    --backend vggt --limit 4 --resolution 256 --n-gt-points 5000 --seed 0

# Crude CPU fallback (no GPU; uncalibrated) — reproduces the RESULTS.md numbers:
python scripts/run_dmc_reconstruction_real.py \
    --data data/poseidon3d/extracted/data \
    --output outputs/dmc_reconstruction_real \
    --backend open3d --limit 4 --resolution 96 --n-gt-points 1500 --seed 0
```

### End-to-end certificate (real geometry -> certificate)

```bash
# KEYSTONE: real Chamfer/error feeds delta_interval_mm -> decide_surface_change.
# t1 is a known +X displacement of the mesh (--shift-mm); 0.0 => stable pair.
python scripts/run_dmc_certificate_real.py \
    --data data/poseidon3d/extracted/data \
    --output outputs/dmc_certificate_real \
    --backend vggt --limit 8 --shift-mm 1.0 --resolution 256 --seed 0
```

Without the Poseidon3D meshes this script **exits** rather than fabricating a
delta. For an explicitly-labelled synthetic demo (no data/GPU):

```bash
python scripts/run_dmc_certificate_real.py --synthetic --output outputs/dmc_cert_demo
```

### Coverage on real 3D landmarks (3DTeethLand)

```bash
python scripts/run_dmc_teethland.py \
    --data data/teeth3ds/extracted \
    --output outputs/dmc_teethland
```

### Self-consistency Chamfer + coverage (Poseidon3D)

```bash
python scripts/run_dmc_poseidon3d.py \
    --data data/poseidon3d/extracted/data \
    --output outputs/dmc_poseidon3d --seed 0
```

## Status

- The `vggt` / `dust3r` backends are implemented but **pending a GPU run** (they
  need CUDA; without a GPU the chain falls through to the crude Open3D
  fallback). See `RESULTS.md` for the current (Open3D-fallback) numbers and
  `docs/GPU_BASELINE_CONTRACT.md` for the GPU runtime contract.
