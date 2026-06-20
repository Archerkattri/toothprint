# Datasets - DentalChangeCert

All datasets belong under ignored `data/`. Do not commit archives, extracted
images, or patient-level metadata.

## Directory Convention

```text
data/
  <dataset-id>/
    raw/
    extracted/
    manifests/
```

## Priority Sources

| Dataset | Target path | Purpose |
| --- | --- | --- |
| DenPAR | `data/denpar/raw/` | Open periodontal radiograph starting point |
| Periapical lesions | `data/periapical-lesions/raw/` | External periapical robustness |
| Mendeley bitewing caries | `data/mendeley-bitewing-caries/raw/` | Auxiliary dental radiograph domain; check license before release |
| perio-KPT | `data/perio-kpt/raw/` | Landmark uncertainty and CEJ/bone-level supervision |
| PRAD-10K | `data/prad-10k/raw/` | MICCAI periapical segmentation benchmark/domain stress |

## Validation Rules

For each dataset, create a manifest row with:

- dataset ID;
- source URL/DOI;
- license;
- raw archive filename;
- checksum;
- number of images after extraction;
- annotation type;
- patient/case/timepoint fields if present;
- permitted release mode.

## First Extraction Rule

The first extraction script should support a `--limit 10` argument and write a
manifest before any full parse is attempted.

