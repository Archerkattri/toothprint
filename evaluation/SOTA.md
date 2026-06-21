# ToothPrint vs. the state of the art

An honest placement of each mechanism against the published literature (surveyed
2026-06). The recurring theme: ToothPrint is in the *registration / conformal* family,
and its defensible edge is **certification** (finite-sample bounded error) — a lane the
learned-SOTA literature leaves almost entirely empty — not a higher saturated accuracy.

## Identity (3D arch biometrics)

| | Best published | ToothPrint |
|---|---|---|
| Only dedicated 3D-IOS identity pipeline | **Zhou et al. 2024** (Bioengineering): FPFH + SAC-IA + ICP + RMSE — *same family as ours*. Rank-1 **100%**, genuine RMSE 0.198 vs impostor 1.140 mm, on **160 real adults, ~1-yr re-scans**. | Rank-1 **0.995** / EER 0.005 / AUC **0.997** (CI 0.989–0.998) on **all 200** Poseidon3D arches; point-to-surface fidelity **0.05 mm** (sub-noise). |
| Forensic similarity | medRxiv 2025: ICP+RMS, AUC 0.990 (30 subjects). | AUC 0.997 with bootstrap CI. |
| Learned embeddings (PointNet/DGCNN/transformer) for **identity** | **None exist** (only segmentation/landmarks). | open niche — not required to beat. |
| **Certified / conformal identity (bounded false-match rate)** | **None exist in dentistry** (confirmed). | **Novel:** split-conformal accept threshold; empirical FMR tracks target α; **open-set** FNIR 0.030 @ FPIR 1%. |

**Where we lead:** explicit EER/AUC+CI, sub-noise alignment fidelity, and a *certified*
bounded-FMR decision + open-set rejection that no prior dental work reports.
**Where we trail:** Zhou's data is *real* 1-year re-scans on 160 people; ours is
synthetic re-scan augmentation. One hard arch (partial/anomalous) is our only Rank-1 miss
— the known partial-overlap limit of a rigid PCA init.

## Change (radiograph bone-level / periodontal)

| | Best published | ToothPrint |
|---|---|---|
| Cross-sectional measurement | CNN/transformer (Yu 2024 SegFormer RBL r>0.85; Xue 2024 TransUNet 89.45%). Meta-analysis ceiling: **sens 87% / spec 76% / acc 84%**. | not competing on single-timepoint staging. |
| Landmark localization | reported in **px / PCK** (datasets aren't mm-calibrated); 2021 hourglass ~ best. Ours ~36 px is ~3× behind in px. | detector front-end is the honest weak point. |
| **Longitudinal differential change between timepoints** | named **THE open problem** by 3 systematic reviews; prior art is BSI/SIENA (brain), classical DSR (Yi 2005), sub-pixel NCC (Guizar-Sicairos 2008). | differential sub-pixel registration + multi-anchor affine; **measurement recall 0.98 @ true 0% false-progression**. |
| **Conformal false-progression certificate** | **None** (change×dental×conformal is empty). | **Novel:** false-progression bounded ≤ α, distribution-free. |

**Where we lead:** the differential + conformal-certificate combination is unoccupied.
**Where we trail:** the automatic detector (~36 px) caps end-to-end recall at 0.81 — a
label-resolution limit, not a certificate flaw. P0 improvement: HRNet-W32 / YOLOv8-pose.

## Surface (3D change certificate)

| | Best published | ToothPrint |
|---|---|---|
| Cloud-to-cloud change w/ confidence | **M3C2** (Lague 2013) — geomorphology standard, normal-direction distance + CI; C2C/C2M weaker. | de-biased differential displacement + **regional** max statistic. |
| Localized lesion vs whole-surface average | — | regional recall **0.99** vs **0.00** global. |
| **Conformal false-change bound** | rare in imaging change-detection. | false-change rate ≤ α by max-calibration. |

**Benchmarked vs M3C2** (a 0.5 mm lesion over 2 % of the arch, recall @ 5 % false-change,
noise swept 0.1→0.6 mm): the **whole-surface average collapses** (1.0→0.10 — the dilution
failure mode), while **M3C2 (1.0→0.50) and ours (1.0→0.17) both localize it**. Honest
verdict: M3C2 is a strong localized baseline and edges us on *raw recall* at extreme noise;
we do **not** claim to beat it there. Our genuine, complementary edge is the **finite-sample
conformal false-change bound** (M3C2 reports a distance, not a calibrated decision with a
guaranteed FPR) plus the de-biasing that holds usable noise to ~0.4 mm. Both crush the
whole-surface average — which is the real lesson: localized + certified, not a global mean.

## Reconstruction (photos → mesh)

SOTA GS→mesh (2DGS, GOF, PGSR) reports sub-mm Chamfer on DTU/T&T. Ours: 3DGS + multi-view
TSDF fusion → 1.2 M-tri watertight mesh, **0.42 mm median / 0.54 mm Chamfer** at arch
scale — usable for the surface certificate's de-biased edge, though an IOS scan is still
better. Next step: 2DGS rasterization / GOF-style opacity-field meshing for flatter,
normal-consistent surfaces.

## Bottom line

ToothPrint's accuracy is competitive with the (thin, registration-family) SOTA on its
saturated metrics, and it is **ahead on the axis nobody else occupies: a finite-sample,
distribution-free *certificate* on every verdict** — identity FMR, change false-progression,
surface false-change. The honest gaps are real longitudinal data and the radiograph
detector front-end, both of which are non-code (data/labels), not method, limits.
