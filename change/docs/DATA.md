# Data: download, extract, and reproduce `outputs/`

This document records the **exact** steps to put each dataset into the paths the
gate scripts expect, and the exact commands that regenerate everything under
`outputs/`. Nothing here downloads data automatically — DenPAR is open but the
archive must be present locally, and perio-KPT is access-gated.

All data lives under the git-ignored `data/` tree. Do **not** commit archives,
extracted images, or patient-level metadata.

## TL;DR

```bash
# 1. verify the raw DenPAR archive and extract it into the expected path
scripts/fetch_data.sh verify      # checksum-only
scripts/fetch_data.sh denpar      # extract -> data/denpar/extracted/Dataset/

# 2. (optional) place + extract the access-gated perio-KPT dataset, then:
scripts/fetch_data.sh perio-kpt   # prints the required layout

# 3. regenerate outputs/ (see "Reproducing outputs/" below)
```

`scripts/fetch_data.sh all` runs both steps. It is idempotent; pass `--force`
to re-extract over an existing tree.

---

## DenPAR (core, open — CC-BY-4.0)

| Field | Value |
| --- | --- |
| Role | core dataset for Gate-2 and the M4 perturbation-shift experiment |
| License | CC-BY-4.0 (redistribution allowed) |
| DOI | 10.5281/zenodo.16645076 |
| Paper | https://www.nature.com/articles/s41597-025-05906-9 |
| Raw archive | `data/denpar/raw/DenPAR_Radiographs_Dataset.zip` |
| SHA-256 | `b9edb55020f2cb971ba771b4cf5e4b65c4abb4df957310bd2eccc83d5a08b072` |
| Extracted path | `data/denpar/extracted/Dataset/` |

### Download (manual — the archive ships in `data/denpar/raw/` already)

If the raw zip is missing, fetch it from the DOI above and place it at
`data/denpar/raw/DenPAR_Radiographs_Dataset.zip`, then verify:

```bash
sha256sum data/denpar/raw/DenPAR_Radiographs_Dataset.zip
# expect: b9edb55020f2cb971ba771b4cf5e4b65c4abb4df957310bd2eccc83d5a08b072
```

### Extract

```bash
scripts/fetch_data.sh denpar
# equivalent to:
#   mkdir -p data/denpar/extracted
#   unzip -q data/denpar/raw/DenPAR_Radiographs_Dataset.zip -d data/denpar/extracted
```

The archive unpacks to a single `Dataset/` directory, giving the layout the
`RealDenparAdapter` expects:

```text
data/denpar/extracted/Dataset/
  Training/   { Images/, Key Points Annotations/, Bone Level Annotations/ }
  Validation/ { Images/, Key Points Annotations/, Bone Level Annotations/ }
  Testing/    { Images/, Key Points Annotations/, Bone Level Annotations/ }
  Characteristics of radiographs included.xlsx
```

`Training/` -> `train`, `Validation/` -> `val`, `Testing/` -> `test`.

---

## perio-KPT (landmark supervision — access-gated)

| Field | Value |
| --- | --- |
| Role | second real source for Gate-2 (perio-KPT) and M4 cross-source shift |
| Access | request-access / gated — **cannot be auto-downloaded** |
| Annotations | CEJ, alveolar-crest (bone-level), and root-apex keypoints on IOPA images |
| Raw archive | place under `data/perio-kpt/raw/` |
| Extracted path | `data/perio-kpt/extracted/perio_KPT/` |

perio-KPT is **not present in this repo** and is access-gated, so there is no
automatic download. To use it:

1. Obtain the perio-KPT archive through its access process.
2. Place the archive under `data/perio-kpt/raw/`.
3. Extract so the tree the `PerioKptAdapter` expects is produced:

```text
data/perio-kpt/extracted/perio_KPT/
  0_Baseline/                         { images/, labels/ }
  1_Experiment/standard_box/
      f0/train/{images,labels}/  ...  (YOLO-keypoint folds)
      f0/val/{images,labels}/    ...
      holdout_test_standard_box/{images,labels}/
  3_External_Set/standard_box/        { images/, labels/ }
```

Splits map as: `0_Baseline` -> `baseline`, `1_Experiment` -> `experiment`,
`holdout_test_standard_box` -> `holdout`, `3_External_Set` -> `external`.

Until perio-KPT is present, run only the DenPAR gate; the perio-KPT gate and the
M4 cross-source experiment will (correctly) refuse to run on missing real data.

---

## Other datasets (not on the certificate's critical path)

These ship as raw archives for auxiliary/robustness work and are documented in
`outputs/dataset_manifest.json`:

| Dataset | Raw archive | License | Role |
| --- | --- | --- | --- |
| Periapical lesions | `data/periapical-lesions/raw/periapical_lesions_radiographs.zip` | CC-BY-4.0 | external periapical domain shift |
| Mendeley bitewing caries | `data/mendeley-bitewing-caries/raw/mendeley_bitewing_caries_4fbdxs7s7w_v1.zip` | CC-BY-NC-3.0 | **non-commercial — keep out of any permissive release** |

---

## Reproducing `outputs/`

The numbers in `RESULTS.md` come from the commands below. Run them **after**
the datasets are extracted to the paths above. None of these have been run in
the "make-it-real" pass — they require a GPU/data machine.

### 0. Environment

```bash
pip install -e ".[dev]"            # core + pytest
pip install -e ".[detector]"       # torch + torchvision (for the detector path)
pip install openpyxl               # periapical-lesions adapter only
```

### 1. (Optional but recommended) train the real landmark detector

Produces the `.pt` checkpoint the gates consume via `--detector-weights`.
PENDING a GPU run — no checkpoint exists yet.

```bash
python scripts/train_landmark_detector.py \
    --data data/denpar/extracted/Dataset \
    --output outputs/landmark_detector \
    --epochs 30 --batch-size 4 --device cuda
# -> outputs/landmark_detector/checkpoint_best.pt  (+ train_log.json with
#    per-landmark val pixel error)
```

### 2. Gate-2 on DenPAR

The gate will **not** silently use synthetic landmarks; choose exactly one of
`--detector-weights` (real) or `--synthetic-landmarks` (explicit stand-in).

```bash
# real predicted landmarks (after step 1)
python scripts/run_gate2_denpar.py \
    --data data/denpar/extracted/Dataset \
    --output outputs/gate2_denpar \
    --detector-weights outputs/landmark_detector/checkpoint_best.pt \
    --alpha 0.1

# explicit synthetic stand-in (noisy GT) — labelled as synthetic in metrics.json
python scripts/run_gate2_denpar.py \
    --data data/denpar/extracted/Dataset \
    --output outputs/gate2_denpar \
    --synthetic-landmarks --alpha 0.1
```

### 3. Gate-2 on perio-KPT (requires the gated dataset)

```bash
python scripts/run_gate2.py \
    --data data/perio-kpt/extracted/perio_KPT \
    --output outputs/gate2 \
    --detector-weights outputs/landmark_detector/checkpoint_best.pt
#   ... or --synthetic-landmarks
```

### 4. M4 covariate-shift evaluation (real data required by default)

```bash
python scripts/run_m4_shift_eval.py \
    --denpar-root data/denpar/extracted/Dataset \
    --perio-kpt-root data/perio-kpt/extracted/perio_KPT \
    --alpha 0.1
# Missing real data exits non-zero. Use --allow-synthetic ONLY to run the
# synthetic fallback, whose numbers are NOT a real-data result.
```

### 5. Failure gallery (from a completed gate run)

```bash
python scripts/failure_gallery.py --input outputs/gate2_denpar
```

## Split discipline (do not skip)

Splits are frozen by `dcc/data/split_registry.py` (SHA-256). Calibrate on the
train split only, tune `tau` on val only, and report final numbers on test in a
single evaluation with no re-tuning. See `REPRODUCIBILITY.md`.
