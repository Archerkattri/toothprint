# Real Results (from real data)

These numbers come from running the actual pipeline on real datasets — no
synthetic / LCG fallback. Output JSON lives under `outputs/` (gitignored);
reproduce with the commands below.

## Headline — the surface-change certificate works (real Poseidon3D meshes)

The core contribution is a surface-change *certificate* robust to reconstruction
error. Evaluated on 6 real Poseidon3D dental surfaces with accurate (oracle)
geometry — isolating the decision rule from the reconstruction front-end —
IOS-class reconstruction noise (σ = 0.05 mm) absorbed by the conformal error
radius, and a controlled metric surface change as the true-positive signal:

```bash
python scripts/run_dmc_certificate_oracle.py --data data/poseidon3d/extracted/data \
    --output outputs/dmc_cert_oracle --recon-noise-mm 0.05
```

Calibrated error radius **0.114 mm**. Decisions vs the true surface change:

| true change (mm) | change-certified | stable-certified | false-change rate |
|---:|---:|---:|---:|
| 0.0 | 0.00 | **1.00** | 0.000 |
| 0.2 | 0.00 | **1.00** | 0.000 |
| 0.4–0.8 | 0.00 | 0.00 | 0.000 (uncertain, gray zone) |
| 1.0 | **1.00** | 0.00 | — |
| 1.5 | **1.00** | 0.00 | — |

**The certificate certifies stability for ≤0.2 mm, certifies change for ≥1.0 mm,
and abstains in the 0.35–0.75 mm gray zone — all at a 0% false-change rate.**
This is the certificate doing exactly its job on real dental geometry.

![DMC oracle certificate](docs/dmc_oracle_certificate.png)

## Real neural reconstruction backend (DUSt3R) runs on consumer GPU

`--backend dust3r` now runs end-to-end on an 8 GB RTX 4060 (mini-dust3r,
512-px), reconstructing ~104 k points from the 5 protocol views. Surface error
uses **similarity (scale-aware) alignment** (`estimate_scale=True`) because
multiview neural backends recover geometry up to an unknown global scale.
Reconstruction *accuracy* where the views cover the surface is sub-mm; full-arch
Chamfer is coverage-limited (5 views do not see occluded/posterior regions —
an evaluation-protocol limit, not a backend-quality one). VGGT remains the
higher-VRAM option for the 32 GB target hardware.

## Screened-Poisson surface refinement (real, reproducible, CPU)

The crude Open3D edge-projection backend has been **deleted** (no fallbacks), so
absolute end-to-end reconstruction error now requires a real neural backend
(VGGT/DUSt3R) on a GPU with sufficient VRAM — implemented but pending hardware.
What is fully reproducible today (CPU only) is the surface-refinement and
surface-error machinery the certificate's error radius depends on.

Screened Poisson (Kazhdan & Hoppe 2013), validated on a real sub-mm Poseidon3D
IOS arch (Chamfer vs a 200k GT sample, depth 9):

| reconstruction noise sigma | raw Chamfer | after Poisson | effect |
|---|---|---|---|
| 0.0 mm (clean IOS) | 0.162 mm | 0.133 mm | −18% (helps) |
| 0.5 mm (good neural recon) | 0.379 mm | 0.217 mm | −43% (helps a lot) |
| 1.0 mm (decent recon) | 0.598 mm | 0.404 mm | −32% (helps) |
| 2.0 mm (mediocre recon) | 1.006 mm | 1.146 mm | +14% (HURTS) |

So Poisson is a real quality win **only in the sub-mm..~1 mm regime**; past ~2 mm
it over-smooths. See `docs/dmc_poisson_refinement.png` and
`docs/dmc_chamfer_vs_noise.png`.

```bash
python scripts/make_dmc_plots.py   # regenerate the Poisson + surface-error plots
```

> **Backend note:** the metric reconstruction backends (`--backend vggt` default,
> or `dust3r`) are implemented but need a GPU with enough VRAM; on insufficient
> VRAM they **raise** rather than fall back to a crude path. There is no longer
> any CPU edge-projection backend.

## Coverage on real 3D landmarks — 3DTeethLand

Real voxel-grid coverage over all 240 real landmark clouds (fixed grid for
comparability):

```bash
python scripts/run_dmc_teethland.py --output outputs/dmc_teethland
```

Coverage fraction mean ≈ 0.25 (real, method="reconstruction", no synthetic
fallback). See `outputs/dmc_teethland/teethland_coverage.json`.

## Self-consistency Chamfer — Poseidon3D

```bash
python scripts/run_dmc_poseidon3d.py --output outputs/dmc_poseidon3d --seed 0
```

Real per-mesh coverage and a self-consistency Chamfer (~0.8 mm) between two
independent samples of the same real mesh.
