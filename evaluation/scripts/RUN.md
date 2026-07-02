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

Sonata needs `spconv`, `torch-scatter`, `timm`, `addict`, `huggingface_hub` (Flash-Attention is
optional). **Verified working 2026-07-02 on an RTX 5090 (CUDA 12.8 / sm_120, torch 2.11+cu128):**

```bash
# spconv: no cu128 wheel yet, but the cu126 wheel's cumm backend JIT-compiles kernels for the
# actual GPU arch at runtime, so it works on Blackwell (sm_120):
pip install spconv-cu126
# torch-scatter: no prebuilt wheel for torch 2.11 — build from source against the local toolkit:
CUDA_HOME=/usr/local/cuda FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=12 \
  pip install --no-build-isolation --no-cache-dir torch-scatter
pip install timm nibabel                       # timm.layers.DropPath; nibabel is for BUFFER-X below
pip install --no-deps --no-build-isolation 'git+https://github.com/facebookresearch/sonata.git'
# addict / huggingface_hub are usually already present; weights 'facebook/sonata' (public, ungated)
# download on first .load(). Flash-Attention is NOT required (see enable_flash_attn below).
```

Sanity-check the backbone loads + forwards (downloads weights on first call):

```bash
python -c "import sonata; m = sonata.load('sonata', repo_id='facebook/sonata', custom_config={'enable_flash': False}); print(type(m).__name__)"
```

The `SonataEmbedding` wrapper handles the two integration details automatically: it loads with
`enable_flash_attn=False` (standard serialized-attention path — no Flash-Attention wheel needed)
and feeds the 9-channel input the pretrained PTv3 stem expects (`coord` + zero colour + per-point
PCA-estimated normals). Install Flash-Attention and pass `enable_flash_attn=True` only if you want
the speedup.

### Train

```bash
# Teeth3DS+ (real, .obj arches), frozen encoder, head-only (recommended for a small subject set).
# TP_NTRAIN reserves the rest as an unseen held-out split; the script writes sonata_identity.json.
TP_DATA=$TP_TEETH3DS TP_NTRAIN=110 TP_EPOCHS=80 TP_FREEZE=1 \
  python evaluation/scripts/train_sonata_embedding.py

# full fine-tune (needs more VRAM; use if frozen underperforms — it did on Teeth3DS, see README)
TP_DATA=$TP_TEETH3DS TP_NTRAIN=110 TP_EPOCHS=80 TP_FREEZE=0 TP_GRID=0.02 \
  python evaluation/scripts/train_sonata_embedding.py
```

Weights save to `TP_SONATA_WEIGHTS` (default `/tmp/toothprint_embedding/sonata_encoder.pt`,
gitignored). Env knobs: `TP_NTRAIN` (train subjects; rest held out), `TP_FREEZE` (1/0), `TP_GRID`
(PTv3 voxel grid, default 0.02 on the unit-normalised arch), `TP_SONATA_REPO` (weights repo),
`TP_KEEP_LO` (crop floor, inherited). Poseidon3D (`.stl`) still works via the same script.

**Measured (2026-07-02, frozen head-only, 110 train / 40 held-out real Teeth3DS+):** held-out
Rank-1 **0.275 / 0.125 / 0.025** at keep 1.0 / 0.5 / 0.3 — an honest **negative** (train-acc
plateaus ~0.23): frozen indoor-SSL features do not transfer to dental identity in this low-data,
head-only recipe. `TP_FREEZE=0` full fine-tune is the open next step.

### Expected GPU / memory

| mode              | GPU (min)      | peak VRAM (batch=16) | wall-clock (80 ep, 150 subj) |
| ----------------- | -------------- | -------------------- | ---------------------------- |
| frozen head-only  | 12 GB (e.g. 3060) | ~6–8 GB           | ~1–2 h                       |
| full fine-tune    | 24 GB (e.g. 3090/4090) | ~16–22 GB    | ~4–6 h                       |

PTv3 is heavier than DGCNN, so batch is 16 (vs 32). If you OOM, halve `BATCH` in the script or
lower `N_PTS`. These are estimates on a single modern consumer GPU; confirm on your hardware.

### Evaluate

`train_sonata_embedding.py` runs its own held-out Rank-1 identity eval after training (keep 1.0 /
0.5 / 0.3, same query synthesis as `eval_embedding.py`) and writes `sonata_identity.json` — no
separate command needed. (`eval_embedding.py` / `eval_embedding_partial.py` are DGCNN-specific;
they instantiate `DGCNN` directly, so they do not load the Sonata checkpoint as-is.)

---

## 2. BUFFER-X zero-shot registration baseline (`eval_bufferx_baseline.py`)

Benchmarks the generalist **BUFFER-X** (ICCV 2025, arXiv 2503.07940) against our dental-specialised
**CorrNet** on the *identical* partial-overlap crop protocol (keep-0.5 / keep-0.3, planar +
realistic whole-tooth dropout, Rank-1), **on real Teeth3DS+ arches**. Reference: CorrNet ~0.87 /
~0.57 and rigid GICP 0.23 / 0.10 (both recorded on Poseidon3D).

### Install

`eval_bufferx_baseline.py` loads the pretrained BUFFER-X **3DMatch** model from an already-built
BUFFER-X tree (`snapshot/threedmatch/{Desc,Pose}/best.pth` + built `cpp_wrappers/*.so`), the loader
mirroring the working `_load_bufferx` in the splatreg codebase. Point `BUFFERX_REPO` at that tree:

```bash
# a built BUFFER-X checkout (repo README's compile_wrappers.sh + HF snapshot download); needs a
# venv with torch(+CUDA), pointnet2_ops, knn_cuda, kornia, nibabel, open3d.
export BUFFERX_REPO=/abs/path/to/BUFFER-X
```

### Run

```bash
# real Teeth3DS+ upper arches; N arches (gallery == probe set), dense point budget for BUFFER-X.
TP_TEETH3DS=$TP_TEETH3DS BUFFERX_REPO=$BUFFERX_REPO TP_BUFFERX_N=40 TP_BUFFERX_NP=8000 \
  python evaluation/scripts/eval_bufferx_baseline.py
```

Env knobs: `TP_BUFFERX_N` (arch subset), `TP_BUFFERX_NP` (points/arch — keep-0.3 crops must stay
above ~2000 for BUFFER-X's FPS), `TP_BUFFERX_REPS`, `TP_BUFFERX_MODES`, `TP_BUFFERX_KEEPS`. Writes
`evaluation/results/bufferx_baseline.json`. If the tree / weights / CUDA extensions are missing the
script prints the setup and exits non-zero (no fake numbers).

**Measured (2026-07-02, N=40 real Teeth3DS+, NP=8000):** BUFFER-X zero-shot Rank-1 **1.00 / 0.95**
at keep-0.5 / keep-0.3 (realistic dropout; planar 1.00 / 1.00) — zero-shot registration transfers
to teeth (~22 min on one RTX 5090). See the README for the head-to-head table and caveats.

### Expected GPU / memory

BUFFER-X inference only (no training): a few hundred MB–GB VRAM, ~0.2–0.9 s per registration at
NP≈6k–8k on an RTX 5090. The gallery matrix is `n×n` registrations per (mode, keep, rep); N=40 with
two modes × two keeps × 1 rep ≈ 6.4k registrations, ~22 min.

### API integration note

The single upstream touch-point is `_load_bufferx()` in `eval_bufferx_baseline.py`: it builds the
3DMatch config, loads the Desc then Pose checkpoints (both full-model state_dicts) into the whole
model with `strict=False`, and wraps the model's test-mode forward (sphericity-based voxel
downsample → `(R, t, score)`). If the upstream layout drifts, adapt only that function — the
protocol/scoring around it is fixed and shared with `eval_correspondence.py`.
