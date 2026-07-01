# Reconstruction leg — 2026 SOTA upgrade notes

Status snapshot from the 2026-07-01 SOTA review of the reconstruction front-end (the leg that
turns intraoral photos into the mesh the surface/change certificates measure). The certification
layer is untouched; this is only about the geometry front-end that feeds it.

## Current front-end (the baseline to beat)

**2D Gaussian Splatting (oriented surfels) + multi-view TSDF fusion.** 2DGS disks lie *on* the
surface, so meshing from the **median** depth (first-surface crossing, not the alpha-weighted mean
that averages an arch's front and back walls) is markedly sharper than 3DGS.

- **~0.264 mm median** on the reference arch, **~0.3 mm median** across the n=5 gallery
  (`evaluation/results/reconstruction.json`, `make_2dgs_mesh.py`).
- **38.9% better than 3DGS+TSDF** median-for-median, up to 2.4× on the hardest arch.
- Output: watertight ~1 M-triangle mesh, no scanner required.

This is the number every candidate below must be measured against, on the same per-arch
median/mean/chamfer-vs-GT-IOS protocol.

## Candidates to evaluate (intraoral floater / specular robustness)

Intraoral capture is dominated by specular highlights (wet enamel), thin floaters, and narrow
baselines — the failure modes these newer surfel/hybrid splatters target. Evaluate each as a
drop-in replacement for the 2DGS front-end, same TSDF meshing, same GT comparison:

| Candidate | Venue | Why it might help here |
| --- | --- | --- |
| **MeshSplat** | AAAI 2026 | Splatting that regresses a mesh directly — could remove the TSDF step and tighten thin-structure recovery. |
| **SurfelSplat** | 2026 | Surfel-native splatting; potentially better first-surface consistency than disk-2DGS on curved enamel. |
| **Gaussian-Voxel Duet** | 2026 | Hybrid Gaussian + voxel representation for floater suppression and specular robustness under narrow baselines. |

Evaluation gate: none adopted unless it beats 0.264 mm median (reference arch) / ~0.3 mm gallery
median **without** regressing the watertightness the certificates require.

## Dental-recon competitors that exist but do NOT certify

For positioning. These are the 2024–2026 dental 3D-reconstruction systems in the literature. None
report **mm-level accuracy vs a ground-truth IOS scan** the way we do (0.264 mm median), and
**none produce a certificate** (conformal or otherwise) on the reconstructed geometry:

| System | Ref | Gap vs ToothPrint |
| --- | --- | --- |
| **DentalSplat** | arXiv 2511.03099 | Gaussian-splat dental recon; no mm-level GT-IOS accuracy, no certificate. |
| **Dental3R** | arXiv 2511.14315 | Dental 3D recon; no certified error bound. |
| **DentalGS** | AAAI 2026 | Gaussian-splat dental recon; no mm accuracy vs IOS, no certificate. |
| **TeethDreamer** | MICCAI 2024 | Generative teeth reconstruction; qualitative, no certified metric geometry. |
| **DentalMonitoring** | FDA DEN230035 | Cleared monitoring product; proprietary, no open mm accuracy, no conformal certificate. |

**Takeaway:** the reconstruction leg is competitive on raw accuracy (0.264 mm median, 38.9% over
3DGS) and is the *only* one that hands its output to a certified error/change layer. The 2026
upgrade opportunity is robustness (floaters/specular) via MeshSplat / SurfelSplat / Gaussian-Voxel
Duet — a front-end swap that must clear the same accuracy + watertightness bar before adoption.
