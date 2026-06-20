# Agent Instructions - DentalChangeCert

You are working on DentalChangeCert, a research repo for certified intraoral
dental radiograph change detection under acquisition uncertainty.

## Read First

1. `README.md`
2. `docs/dataset-risk-note.md`
3. `dcc/cli.py`
4. `tests/`

## Guardrails

- Do not commit raw data, extracted images, patient data, tokens, `.env`, or
  `LOCAL_SECRETS.md`.
- Keep data under ignored `data/`.
- Keep generated reports under ignored `outputs/`.
- Treat this as the GPU development path and expose CUDA device 0 by default.
- Do not overclaim clinical validity.
- Do not treat synthetic demo outputs as experiments.

## First GPU Tasks

1. Install with `python -m pip install -e .`.
2. Run `python -m pytest tests/ -q --cov=dcc --cov-report=term-missing`.
3. Place raw archives under `data/`.
4. Add extraction/manifest scripts one dataset at a time.
5. Add tests that use tiny fixtures before touching full archives.

## Research Positioning

The novelty is the certificate: robust longitudinal change or abstention under
acquisition/landmark uncertainty. Periapical lesion segmentation, keypoint
detection, and panoramic progression are related work, not the central claim.
