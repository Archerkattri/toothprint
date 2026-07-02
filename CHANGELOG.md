# Changelog

All notable changes to ToothPrint are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0/).

> Full method, results, and honest limitations live in the **[README](README.md)**;
> the peer-facing write-up is the engrXiv preprint [10.31224/7403](https://doi.org/10.31224/7403).

## [Unreleased]

The 2026-07-01 SOTA-upgrade scaffolding was **run on the real Teeth3DS+ data acquired 2026-07-02**
(150 ungated, md5-verified OSF upper arches — single-timepoint, so still not the gate-#7
longitudinal validation).

### Added

- **BUFFER-X zero-shot registration — measured on real arches, now with error bars and a
  complete identity column.** Ran `eval_bufferx_baseline.py` (rewired to load the pretrained
  BUFFER-X 3DMatch model from the built third-party tree and to run on real Teeth3DS+ arches at
  the CorrNet crop protocol). BUFFER-X, an indoor-scan generalist with **no dental training**,
  reaches Rank-1 **1.00 / 0.95** at keep-0.5 / keep-0.3 (realistic whole-tooth dropout; planar
  1.00 / 1.00), on N=40 arches over **3 crop-seed reps**: keep-0.5 held 1.00 in all 3 reps,
  keep-0.3 was 0.95 ± 0.04 (min–max 0.90–0.98) — the single-rep headline holds, now with error
  bars. Added `eval_bufferx_identity_full.py`, which runs the **full-coverage** identity protocol
  (the one the PCA-init+GICP smoke uses) with BUFFER-X as the registration/scoring backend and the
  identical `eval_id3d.metrics` definitions: **Rank-1 1.000 · Rank-5 1.000 · EER 0.000 · AUC
  1.000** (N=40, `evaluation/results/bufferx_identity_full.json`), matching the GICP smoke on every
  identity metric. Zero-shot registration transfers to dental micro-geometry. BUFFER-X is an
  **optional** partial-overlap registrar — the certified pipeline's defaults (PCA-init + GICP +
  CorrNet, conformal accept/abstain) are unchanged. Results in
  `evaluation/results/bufferx_baseline.json` (+ per-rep values, AUC ranges) and
  `bufferx_identity_full.json`; whiskered chart `docs/partial_overlap_results.png`; table + honest
  cross-dataset caveats in the README; option pointer in `evaluation/scripts/RUN.md`.
- **Sonata/PTv3 foundation-model embedding — installed, run, honest negative.** The Pointcept
  stack (spconv-cu126, torch-scatter built from source, `sonata`) installs and runs on the RTX
  5090 (CUDA 12.8 / sm_120); Flash-Attention is optional and off by default. Trained a **frozen**
  Sonata encoder + ArcFace head on 110 real arches: held-out Rank-1 **0.275 / 0.125 / 0.025** at
  keep 1.0 / 0.5 / 0.3 on 40 unseen subjects (`evaluation/results/sonata_identity.json`) — well
  below the from-scratch DGCNN (0.995 full-coverage, Poseidon3D) and the rigid pipeline (1.0 on
  Teeth3DS+). Frozen indoor-SSL features do not transfer to dental identity in this low-data,
  head-only recipe; full fine-tune (`TP_FREEZE=0`) is the open next step.

### Fixed

- **`SonataEmbedding` backbone made runnable.** The speculative wrapper could not load the
  pretrained weights or forward: (1) it did not disable Flash-Attention (added `enable_flash_attn`,
  default off, passed as `custom_config` to `sonata.load`); (2) it fed 3 input channels where the
  pretrained PTv3 stem expects 9 (`coord + colour + normal`) — now emits zero colour and
  per-point PCA-estimated normals (autocast-safe, float32 `eigh`); (3) the lazily-built backbone
  stranded on CPU when `.load()` followed `.to(device)` — now follows the head's device.
- **Sonata training script generalised to Teeth3DS+** — `train_sonata_embedding.py` now globs
  `.obj` as well as `.stl`, takes a configurable train/held-out split (`TP_NTRAIN`), and runs a
  held-out Rank-1 identity eval (keep 1.0 / 0.5 / 0.3), writing `sonata_identity.json`.

## [1.1.0] - 2026-07-02

Groundwork from a 2026-07-01 SOTA review — code and data-access scaffolding, committed but
**not yet run at scale on this machine, so there are no new headline numbers.** The binding
constraint stays the same: real longitudinal / cross-session data (gate #7), still open.

### Added

- **Foundation-model embedding option** — `backbone="sonata"` in
  `toothprint/identity/embedding.py` adds a PTv3 + Sonata self-supervised encoder (via
  [Pointcept](https://github.com/Pointcept/Pointcept)) behind the same ArcFace head and
  descriptor contract as the from-scratch DGCNN. To our knowledge the first application of a
  point-cloud foundation model to dental identification — stated as a **direction, not a
  result**: it needs a GPU + Pointcept install and has not been trained here for want of real
  dental data. Entry point `evaluation/scripts/train_sonata_embedding.py`; commands and VRAM
  in `evaluation/scripts/RUN.md`.
- **BUFFER-X zero-shot baseline harness** — `evaluation/scripts/eval_bufferx_baseline.py`
  runs BUFFER-X (ICCV 2025) on the identical keep-0.5 / keep-0.3 partial-overlap protocol as
  CorrNet, for a head-to-head table (CorrNet targets recorded in the script header).
- **`detect` optional extra** (`ultralytics>=8.3`) for the YOLO26-pose tooth/landmark
  detector used by the change-evaluation scripts.
- **Reconstruction-upgrade notes** — `docs/RECON_UPGRADES_2026.md`: MeshSplat / SurfelSplat /
  Gaussian-Voxel Duet candidates to beat the ~0.264 mm-median 2DGS+TSDF front-end, and the
  DentalSplat / Dental3R / DentalGS / TeethDreamer / DentalMonitoring systems that neither
  report mm-level GT accuracy nor certify.
- **Real-data gate tracker** — `evaluation/DATA_GATE.md` tracks gate #7 (longitudinal /
  cross-session) with a Zenodo 11392406 DUA application checklist and the credentialed
  PhysioNet Multimodal Dental v1.1.0 path.
- **Teeth3DS+ ungated acquisition + N=40 real-arch identity reproduction** — the repo
  identity pipeline (PCA-init + multi-scale Generalized-ICP, scored by post-alignment surface
  distance) run on real Teeth3DS+ upper arches: **Rank-1 1.000, EER 0.000, AUC 1.000, d′ 6.07**
  (N=40, freshly re-downloaded md5-verified OSF data;
  `evaluation/results/teeth3ds_identity_smoke_n40.json`). Real arches — but **single-timepoint**
  data with synthetic genuine re-scans, so **not** the gate-#7 longitudinal validation.

### Changed

- Test suite grown from **102 → 183 passing** (identity, change, surface, io, api, clinical,
  geometry, embedding).
- README: engrXiv preprint + DOI badges and citation added; the `paper/` folder and the
  Methods-PDF label were dropped in favour of the preprint
  ([10.31224/7403](https://doi.org/10.31224/7403)) and `PAPER.md`.

## [1.0.0] - 2026-06-21

Initial public release of the consolidated ToothPrint system: three certified reads of one
durable dental signal — **identity** (person ID from 3D intraoral scans or 2D radiographs),
**change** (certified longitudinal bone-level change via YOLO26-pose landmarks + differential
sub-pixel registration), and **surface** (certified 3D surface change from a 2DGS + TSDF
reconstruction front-end). Every verdict is **conformal**: it fires only when the interval
around the measurement lies entirely past the threshold, so the false-alarm rate is bounded by
α in finite samples — a distribution-free certificate, or an explicit abstention, never a bare
prediction; the certification core runs without a GPU. Ships as one Python package (identity /
change / surface / io / clinical), a hardened FastAPI service with safe medical-file ingest, and
the cross-platform ToothPrint Studio desktop app with PDF case-report export. Validated on
public single-timepoint data (Poseidon3D, Teeth3DS+, DenPAR, Figshare CBCT+IOS) with synthetic
re-scans/crops — headline numbers are in-simulation ceilings pending real cross-session
longitudinal data. Licensed under PolyForm Noncommercial 1.0.0 (free for any non-commercial use,
including hospitals, clinics, and forensic labs; no resale).
