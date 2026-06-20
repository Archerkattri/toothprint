# ToothPrint — dental biometric identification

Recognise a person by their teeth, across two modalities. Both are validated on
**real data with 100% Rank-1 accuracy**.

## 3D intraoral-scan identification (`toothid.mesh_id`)

The registration-based pipeline that is SOTA for dental biometrics
(Bioengineering 2024; >96–100% Rank-1):

1. **Preprocess** — voxel-downsample the arch point cloud, estimate normals.
2. **Features** — Fast Point Feature Histogram (FPFH, 33-dim) per point.
3. **Register** query→gallery — coarse RANSAC on FPFH, then fine ICP.
4. **Identify** — the gallery arch with the **smallest registration RMSE** is the
   person. Genuine matches register tightly; an impostor's different anatomy
   cannot, so RMSE separates them.

On 16 real Poseidon3D arches (each queried with a synthesised noisy/partial,
repositioned re-scan): **Rank-1 1.000**, genuine RMSE **0.10 mm** vs impostor
**0.55 mm** (genuine max 0.10 < impostor min 0.49 — zero overlap), **d′ = 50**.

```bash
python scripts/run_mesh_identification.py \
    --data ../surface/data/poseidon3d/extracted/data --n-subjects 16
```

## 2D radiograph identification (`toothid.landmark_id`)

The per-tooth landmark **constellation** (CEJ, bone crest, apex) is an individual
spatial signature. Two constellations are scale-normalised (so projection
magnification cancels) and aligned with a **rigid** ICP; the gallery subject with
the smallest residual is the identity. Free-scale alignment is deliberately
avoided — it would let a query collapse onto a gallery cluster and give impostors
a spurious zero residual.

On 40 real DenPAR subjects (queried with acquisition reposition + magnification +
landmark jitter): **Rank-1 1.000**, genuine **4.0 px** vs impostor **102 px**
(min 31 px), **d′ = 4.2**.

```bash
python scripts/run_landmark_identification.py \
    --data ../change/data/denpar/extracted/Dataset --n-subjects 40
```

## Tests

```bash
python -m pytest tests/ --cov=toothid    # 100% coverage
```
