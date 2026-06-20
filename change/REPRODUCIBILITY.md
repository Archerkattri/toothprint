# DCC Reproducibility Checklist

## Environment

```bash
pip install -e ".[conformal,dev]"  # core + conformal extras
pip install openpyxl               # for periapical lesions adapter
```

## Datasets

Exact download/extract steps and the commands that regenerate `outputs/` live
in [docs/DATA.md](docs/DATA.md). `scripts/fetch_data.sh` verifies the DenPAR
archive checksum and extracts it into the expected path (it does not download).

| Dataset | Path | Download |
|---------|------|----------|
| DenPAR | `data/denpar/extracted/Dataset/` | DOI 10.5281/zenodo.16645076 (CC-BY-4.0) |
| perio-KPT | `data/perio-kpt/extracted/perio_KPT/` | access-gated (see docs/DATA.md) |
| Periapical Lesions | `data/periapical-lesions/extracted/` | Zenodo 10.5281/zenodo.13772918 |

## Split Discipline

Splits are frozen by `dcc/data/split_registry.py` using SHA-256 hash.
**Never tune thresholds on test split.** Order of operations:
1. Freeze splits: `dcc.data.split_registry.split_records(records, salt="dcc_v1")`
2. Calibrate on train split only
3. Tune tau on val split only
4. Report final numbers on test split (single eval, no re-tuning)

## Gate Scripts

```bash
# Gate 2 on DenPAR — choose ONE landmark source (no silent synthetic fallback):
#   --detector-weights PATH   real predicted landmarks (train via
#                             scripts/train_landmark_detector.py first), or
#   --synthetic-landmarks     explicit noisy-GT stand-in
python scripts/run_gate2_denpar.py \
    --data data/denpar/extracted/Dataset \
    --output outputs/gate2_denpar \
    --synthetic-landmarks \
    --alpha 0.1

# Shift evaluation (M4) — real data REQUIRED by default;
# pass --allow-synthetic to opt into the synthetic fallback.
python scripts/run_m4_shift_eval.py \
    --denpar-root data/denpar/extracted/Dataset \
    --perio-kpt-root data/perio-kpt/extracted/perio_KPT

# Phantom validation (M5)
python scripts/validate_phantom.py \
    --data data/phantom/extracted

# Failure gallery
python scripts/failure_gallery.py \
    --input outputs/gate2_denpar
```

## Seeds

The Gate-2 DenPAR pipeline fixes its seed via `PairBuilderConfig(seed=42)`;
the M4 shift script uses fixed per-experiment seeds (0–3). The
`PairBuilderConfig` default is `seed=0`. Asymmetric conformal calibration is
deterministic (no randomness). Split registry uses SHA-256.

## Prior Art Note

The bone-level change score is related to patent US 8,768,036 (Sirona Dental)
which describes automated radiographic bone-level measurement. This codebase
differs in that it uses conformal prediction intervals to certify the *absence*
of clinically significant change with coverage guarantees, rather than measuring
absolute bone levels. The conformal framing (Vovk et al. 2005, Papadopoulos 2002)
is distinct from and postdates that patent.
