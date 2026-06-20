# Datasets - DentalMapCert

All datasets and captures belong under ignored `data/`.

## Directory Convention

```text
data/
  <dataset-id>/
    raw/
    extracted/
    manifests/
```

## Priority Sources

| Source | Target path | Purpose |
| --- | --- | --- |
| Teeth3DS | `data/teeth3ds/raw/` | Reference dental meshes and tooth geometry |
| Dental3R | `data/dental3r/raw/` | Sparse dental reconstruction baseline/reference |
| DentalSplat | `data/dentalsplat/raw/` | Dental Gaussian splatting baseline/reference |
| DentalGS | `data/dentalgs/raw/` | Dental GS related work/baseline |
| TeethDreamer | `data/teethdreamer/raw/` | 3D dental reconstruction related work |
| Project phone captures | `data/phone-captures/raw/` | The most important future dataset |

## Project-Owned Capture Protocol

Minimum first protocol:

- anterior view;
- left buccal view;
- right buccal view;
- upper occlusal view if possible;
- lower occlusal view if possible;
- short video sweep variant;
- lighting condition tag;
- phone model tag;
- blur/glare/occlusion quality tags.

## Manifest Fields

Each case should record:

- case ID;
- subject/session/timepoint IDs;
- jaw;
- source dataset;
- reference mesh path if available;
- capture views;
- surface-region labels;
- quality tags;
- license/release status.

## First Compute Step

Start with manifest validation and a 5-case subset. Do not run full
reconstruction before the manifest and split logic are tested.

