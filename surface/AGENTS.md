# Agent Instructions - DentalMapCert

You are working on DentalMapCert, a research repo for coverage-certified
smartphone oral surface mapping.

## Read First

1. `README.md`
2. `docs/GPU_BASELINE_CONTRACT.md`
3. `docs/DATASETS.md`
4. `src/dentalmapcert/cli.py`
5. `tests/`

## Guardrails

- Do not commit raw captures, patient images, meshes, tokens, `.env`, or
  `LOCAL_SECRETS.md`.
- Keep data under ignored `data/`.
- Keep generated outputs under ignored `outputs/`.
- Treat this as the GPU development path and expose CUDA device 0 by default.
- Do not claim hidden anatomy. The project is about visible surfaces.

## First GPU Tasks

1. Install with `python -m pip install -e .`.
2. Run `python -m pytest`.
3. Add a tiny fixture dataset under tests, not real patient data.
4. Add dataset manifests for meshes, captures, and surface regions.
5. Add one CUDA baseline adapter at a time: COLMAP/DUSt3R/MASt3R/VGGT/3DGS.

## Research Positioning

The novelty is the trust layer: coverage, uncertainty, recapture, and
longitudinal visible-surface change. Do not frame this as generic dental 3D
reconstruction.
