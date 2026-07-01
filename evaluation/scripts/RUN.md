# RUN.md — launch commands for the 2026 SOTA-upgrade eval scripts

Exact, copy-paste commands for the two new benchmarks added in the 2026-07-01 SOTA review:
the **Sonata/PTv3 foundation-model identity backbone** and the **BUFFER-X** zero-shot
registration baseline. Both need a GPU and off-machine data; both fail fast with an install
hint rather than fabricating numbers.

All commands assume the shared eval working dir with `data/poseidon3d` present and the package
installed (`pip install -e '.[recon]'` from the repo root). Point `TP_DATA` / `TP_POSEIDON3D`
at your Poseidon3D copy (`<id>/<id>*.stl`) — see `evaluation/scripts/paths.py`.

---

## 1. Sonata-pretrained PTv3 identity backbone (`train_sonata_embedding.py`)

First application of a point-cloud **foundation model** to dental identity. Same 150-subject
split, same crop-hardened recipe as `train_embedding.py` (DGCNN) — swap only the backbone, so
any delta is attributable to the pretrained encoder.

### Install Pointcept + Sonata weights

Pointcept has heavy CUDA deps (`spconv`, `torch-scatter`, `pointops`); match your CUDA toolkit.

```bash
# CUDA 12.x example — adjust the +cu suffix to your toolkit
pip install spconv-cu120
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install 'git+https://github.com/Pointcept/Pointcept.git#subdirectory=libs/pointops'
pip install sonata-3d       # convenience wheel; or: pip install 'git+https://github.com/Pointcept/Pointcept.git'
# huggingface_hub is pulled in transitively; weights 'facebook/sonata' download on first .load()
huggingface-cli login        # only if the repo is gated for you
```

Sanity-check the backbone loads (downloads ~weights on first call):

```bash
python -c "import sonata; m = sonata.load('sonata', repo_id='facebook/sonata'); print(type(m))"
```

### Train

```bash
# frozen encoder, train only the projection head + ArcFace centres (recommended for 150 subjects)
TP_DATA=data/poseidon3d TP_EPOCHS=80 TP_FREEZE=1 \
  python evaluation/scripts/train_sonata_embedding.py

# full fine-tune (needs more VRAM; use if frozen underperforms DGCNN)
TP_DATA=data/poseidon3d TP_EPOCHS=80 TP_FREEZE=0 TP_GRID=0.02 \
  python evaluation/scripts/train_sonata_embedding.py
```

Weights save to `TP_SONATA_WEIGHTS` (default `/tmp/toothprint_embedding/sonata_encoder.pt`,
gitignored). Env knobs: `TP_FREEZE` (1/0), `TP_GRID` (PTv3 voxel grid, default 0.02 on the
unit-normalised arch), `TP_SONATA_REPO` (weights repo), `TP_KEEP_LO` (crop floor, inherited).

### Expected GPU / memory

| mode              | GPU (min)      | peak VRAM (batch=16) | wall-clock (80 ep, 150 subj) |
| ----------------- | -------------- | -------------------- | ---------------------------- |
| frozen head-only  | 12 GB (e.g. 3060) | ~6–8 GB           | ~1–2 h                       |
| full fine-tune    | 24 GB (e.g. 3090/4090) | ~16–22 GB    | ~4–6 h                       |

PTv3 is heavier than DGCNN, so batch is 16 (vs 32). If you OOM, halve `BATCH` in the script or
lower `N_PTS`. These are estimates on a single modern consumer GPU; confirm on your hardware.

### Evaluate

`eval_embedding.py` / `eval_embedding_partial.py` load whatever encoder checkpoint they are
pointed at; run them against `sonata_encoder.pt` to get the head-to-head vs the DGCNN JSON
(`embedding_identity.json`, `embedding_partial.json`). The descriptor contract is identical
(`(B,N,3) -> (B,256)` unit vectors), so no eval code changes.

---

## 2. BUFFER-X zero-shot registration baseline (`eval_bufferx_baseline.py`)

Benchmarks the generalist **BUFFER-X** (ICCV 2025, arXiv 2503.07940) against our dental-specialised
**CorrNet** on the *identical* partial-overlap protocol (held-out unseen subjects, keep-0.5 /
keep-0.3, planar + realistic whole-tooth dropout, Rank-1). Target to beat: CorrNet ~0.87 / ~0.57
(realistic dropout, keep-0.5 / keep-0.3).

### Install

```bash
git clone https://github.com/MIT-SPARK/BUFFER-X
cd BUFFER-X && pip install -r requirements.txt && pip install -e .
# download the released generalist checkpoint per the repo README (into ./weights)
cd -
export BUFFERX_CKPT=/abs/path/to/bufferx_generalist.pth
```

### Run

```bash
TP_DATA=data/poseidon3d BUFFERX_CKPT=$BUFFERX_CKPT \
  python evaluation/scripts/eval_bufferx_baseline.py
```

Writes `evaluation/results/bufferx_baseline.json` next to `correspondence_identity.json` for a
direct table. If `bufferx` or `BUFFERX_CKPT` is missing the script prints the install steps and
exits non-zero (no fake numbers).

### Expected GPU / memory

BUFFER-X inference only (no training here): ~6–10 GB VRAM, a few seconds per pair. The full
gallery matrix is `n×n×REPS×4` registrations — for `n≈52` held-out subjects and `REPS=3` that is
~32k registrations, ~10–30 min on a single GPU depending on the checkpoint.

### API integration note

The single upstream touch-point is `_load_bufferx()` in `eval_bufferx_baseline.py`, which expects
`bufferx.api.load_model(ckpt)` and `bufferx.api.register(model, src, dst) -> (R, t, score)`. If the
upstream symbol names differ, adapt only that function — the protocol/scoring around it is fixed
and shared with `eval_correspondence.py`.
