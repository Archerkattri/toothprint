# ToothPrint: A Certified, Partial-Overlap-Robust System for Dental Biometric Identification

*Methods report — Krishi Attri. This is an engineering/methods writeup of the ToothPrint
evaluation suite; every number below is reproduced by a committed script and result JSON
(`evaluation/`), and every limitation is stated. All identity validation is on single-timepoint
public data with **synthetic** re-scans/crops; real longitudinal validation remains open (§7).*

## Abstract

The dental arch is an individual "tooth print": crown contours, cusp geometry, the gingival
margin, and the pattern of restorations. We assemble a dental-identity system that (i) recognises
a person from a 3D intraoral scan or a 2D radiograph, (ii) returns a **certified** accept/abstain
decision with a distribution-free false-match-rate bound, and (iii) is **robust to partial
overlap** — the missing-teeth query where prior rigid methods collapse. Our contributions are a
learned point-correspondence matcher that lifts 50%-tooth-loss identification from 0.23 (rigid
GICP) to **0.87**, a conformal open-set decision hardened against look-alikes, a restoration
("dental-work") biometric demonstrated on both CBCT and plain radiographs, and an honest
characterisation — full DET curves, ablations, a cross-dataset domain-gap measurement, and the
operating regimes where the system must abstain. The one limit no method here closes is real
cross-session data.

## 1. Introduction

Forensic and clinical dental identification is classically manual (dental charting) or, recently,
rigid 3D scan registration. Two gaps persist in the dedicated literature: (a) no **certificate** —
methods report Rank-N accuracy but never a finite-sample bound on the false-match rate of an
individual verdict; and (b) **partial overlap** — a query arch missing teeth (extraction, trauma,
field-of-view) breaks the rigid alignment that all registration methods depend on. ToothPrint
targets both, and treats honesty about failure regimes as a first-class deliverable.

## 2. Related work and positioning

The only dedicated 3D-intraoral identity pipeline with proper Rank-1 is **Zhou et al. 2024**
(*Bioengineering*) — FPFH + SAC-IA + ICP + RMSE, the same registration family as our geometric
core, reporting Rank-1 100% on 160 real adults with ~1-year re-scans (closed-set, private data).
We match its saturated accuracy (Rank-1 0.995 / EER 0.5% on 200 arches) and **lead on a certified,
bounded-FMR decision with open-set rejection** that no prior dental work reports, and on **learned
partial-overlap correspondence**, which the registration literature lacks for this domain.
ToothPrint's defensible lane is *certification + partial-overlap robustness*, not a higher
saturated number.

## 3. Methods

**3.1 Geometric identity (3D).** A query re-scan is given its best **rigid** fit to each gallery
arch — PCA principal-axis initialisation (four proper-rotation hypotheses; robust to the
self-similar palate that defeats FPFH) refined by multi-scale Generalised-ICP — and scored by mean
point-to-surface distance. Rigid (no scale) so the score is shape, not pose. `align_rigid`.

**3.2 Learned embedding.** A DGCNN (EdgeConv) encoder with a **sub-centre ArcFace** head maps an
arch to an L2-normalised descriptor; permutation-invariant pooling means a partial arch still
embeds near the whole. Trained on a 150-subject split, evaluated on held-out unseen subjects.
*Crop-hardening:* training with aggressive partial crops (keep ≥ 0.35) makes the descriptor
coverage-robust at no full-coverage cost. `embedding.py`, `train_embedding.py`.

**3.3 Learned point correspondence (CorrNet).** The key partial-overlap result. A DGCNN backbone
emits a **unit descriptor per point**, trained with an InfoNCE correspondence loss — cropping a
canonical point set yields ground-truth point matches for free. At test, a partial query is matched
point-to-point against each gallery arch (mutual nearest-neighbours → weighted Procrustes), and the
residual over **all** query points scores the fit. Genuine half-arches yield dense, rigidly
consistent matches; impostors do not. `CorrNet`, `train_correspondence.py`, `eval_correspondence.py`.

**3.4 Conformal certificate.** Calibrating a decision threshold on held-out genuine/nearest-impostor
scores yields an empirical false-match rate ≤ α, distribution-free and finite-sample. A
look-alike-hardened variant calibrates on each subject's *nearest impostor*. `analyze_identity.py`.

**3.5 Unified decision.** Retrieve by embedding (recall) → verify the shortlist by CorrNet
correspondence (precision) → accept above the conformal threshold, else **abstain**.
`eval_unified.py`.

**3.6 Multimodal & dental-work.** On paired CBCT+IOS, three independent biometrics — IOS crowns,
CBCT bone/root geometry, CBCT restoration cloud (HU > 2500) — are scored and fused. The restoration
pattern also extends to 2D radiographs via a per-tooth local-contrast extractor (a restoration is
the patch far brighter than its own tooth's median). `eval_multimodal_full.py`,
`eval_dentalwork_2d.py`.

## 4. Datasets and protocols

| dataset | modality | use | n |
|---|---|---|---|
| Poseidon3D | intraoral STL | 3D identity, embedding, CorrNet | 200 (150 train / 50 held-out) |
| Teeth3DS+ | intraoral OBJ | cross-dataset generalisation | 150 |
| DenPAR | periapical radiographs + masks | 2D identity, 2D dental-work | 400 / 165 restoration-bearing |
| Figshare CBCT+IOS | paired CBCT + IOS | multimodal, CBCT dental-work | 55 |

All datasets are single-timepoint, so **genuine queries are synthetic** — acquisition
repositioning (rotation/translation), sensor noise, and partial crops (planar or realistic
whole-tooth dropout). This is disclosed on every result and is the central limitation (§7).

## 5. Results

**5.1 Identity (full coverage).** 3D Rank-1 **0.995** (N=200, EER 0.5%, AUC 0.997, fidelity
0.05 mm); 2D Rank-1 **1.000** (N=400, EER 0). Conformal empirical FMR tracks α in every ablation
(change FPR 0.000–0.005, surface 0.000). Full DET curves (`docs/det_curves.png`) — a first for
dental identity — put full-coverage 3D at EER 0.5%.

**5.2 Partial overlap (the contribution).** Rank-1 under tooth loss, held-out unseen subjects:

| method | keep-0.5 | keep-0.3 (70% gone) |
|---|---|---|
| rigid GICP | 0.23 | 0.10 |
| crop-hardened embedding | 0.64 | 0.26 |
| **CorrNet (realistic dropout)** | **0.87** | **0.57** |
| CorrNet (planar crop) | 0.91 | 0.80 |

CorrNet ablation: an **untrained** CorrNet already scores 0.70 at keep-0.5 — the gain is ~half the
correspondence-plus-rigid-verification *architecture* and ~half the *learned* descriptors. Honest
corrections kept: the planar-crop numbers were optimistic vs realistic whole-tooth dropout
(keep-0.3 0.80 → 0.57).

**5.3 Multimodal fusion.** On 55 paired patients: IOS 1.000, CBCT-bone 0.945, dental-work 0.927.
At full quality geometry saturates so equal-weight fusion ties; in a degraded regime the **oracle
bound is 1.000** (modalities complementary), naive equal-weight fusion *hurts* (dilution), and
**quality-weighted** fusion beats the best single (0.867 vs 0.833) — a real but Rank-1-only gain
(its AUC regresses), reported as such.

**5.4 Dental-work on 2D radiographs.** Per-tooth local contrast (global thresholding fails on
over-saturated JPEGs) yields a restoration constellation identifying 165 subjects at Rank-1
**0.91–0.99** (robust to jitter + a dropped restoration; chance 0.006).

**5.5 Unified certified decision.** Full coverage **FNIR@FMR=1% = 0.00**; partial (keep-0.5) 0.74
— best of all methods but does not escape the open-set ceiling (§6), so it abstains.

## 6. Honest limitations (what the numbers do *not* claim)

1. **Open-set collapses under partial overlap.** At keep-0.5 every method's FNIR@FMR=1% is
   0.74–0.87 (vs 0.03 full) — a half-arch fits many gallery arches, so impostor/genuine score tails
   overlap. The certified rejection is a near-full-coverage property; under heavy tooth loss the
   correct behaviour is to **abstain**, not decide.
2. **Cross-dataset domain gap.** CorrNet trained on Poseidon3D and run on Teeth3DS+ drops keep-0.5
   0.87 → **0.42** (still 34× chance, still > GICP). The learned descriptors are partly
   dataset-specific; closing this needs multi-dataset training.
3. **Extreme tooth loss is the hard tail.** keep-0.3 realistic is 0.57, not the planar 0.80.
4. **Fusion gain is Rank-1-only and small** (N=30; AUC regresses under naive weighting).

## 7. The binding constraint: real cross-session data

Every identity number here — CorrNet's 0.87, the 0.995, the Teeth3DS results — rests on
**synthetic** re-scans/crops of single-timepoint arches, because no open dataset pairs the same
person across visits with the needed modality. This is a *data* limit no code closes. Candidate
sources (credentialed/restricted): PhysioNet multi-visit CBCT+radiographs, Zenodo pre/post-ortho
pairs. Until such data validates the pipeline, ToothPrint is **best-by-design and validated in
simulation**, not proven on real longitudinal data — and the repository states this throughout.

## 8. Reproducibility

Datasets are gitignored; paths are env-configurable (`paths.py`), reference baselines are read from
committed artifacts (not pasted), and committed synthetic fixtures run the 3D pipeline end-to-end
with no off-machine data (`smoke_test.py`; `TOOTHPRINT_FIXTURES=1`). See `REPRODUCE.md`.

## 9. Conclusion

ToothPrint is, to our knowledge, the most complete and most partial-overlap-robust **certified**
dental-identity system demonstrable without gated longitudinal data: a learned correspondence
matcher that quadruples rigid performance under 50% tooth loss, a conformal accept/abstain
certificate, a dental-work biometric across CBCT and radiographs, and a discipline of reporting the
regimes where it must decline. The path to a clinical claim runs through real cross-session data,
not more code.

## References

- Zhou et al. (2024), *Bioengineering* — 3D intraoral-scan identification (FPFH+ICP).
- Wang et al. (2019) — Dynamic Graph CNN (EdgeConv) for point clouds.
- Deng et al. (2019/2020) — (Sub-centre) ArcFace.
- Segal et al. (2009) — Generalised-ICP.
- Vovk et al. — Conformal prediction (distribution-free finite-sample guarantees).
- Qin et al. (2022) — GeoTransformer (learned correspondence; the class of method CorrNet draws on).
