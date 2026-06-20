# ToothPrint — thorough evaluation & hospital-readiness assessment

A full, ablated evaluation of all three mechanisms on all available real data
(Poseidon3D intraoral scans, DenPAR radiographs). Raw results are in
`evaluation/results/*.json`; the degradation curves are in
`evaluation/ablation_summary.png`. Reproduce with the `scripts/eval_*.py`
harness in the companion repositories.

> **Bottom line up front: NO — this is not ready for hospital use, and it is not
> "perfect."** It is a strong, honestly-validated *research prototype* with one
> genuinely trustworthy property (a conformal false-alarm guarantee). Every
> headline number is measured against **synthetic perturbations of
> single-timepoint data**, because no real longitudinal or cross-session dental
> data exists in these datasets. Sensitivity also degrades sharply under
> realistic noise. Details and the path forward below.

## Headline metrics (real data + synthetic perturbations)

| Mechanism | Metric | Result | Caveat |
|---|---|---|---|
| Identity — 3D scans (**N=80**) | Rank-1 / EER / AUC | **1.000 / 0.000 / 1.000** | genuine = synthetic re-scan of the *same* scan |
| Identity — 2D radiographs (**N=179**, full test set) | Rank-1 / EER / AUC | **1.000 / 0.000 / 1.000** | robustness is partly by-design invariance |
| Surface certificate (IOS noise) | recall @1 mm / stable ≤0.2 mm / FPR | **1.000 / 1.000 / 0.000** | synthetic uniform displacement |
| Change **measurement** (accurate localization) | recall @ **true 0 %** false-progression | **0.98** (0.95 at 4 px → **1.00** at ≥16 px) | tau = 2 px; FPR = 0.000 |
| Change certificate | false-progression rate vs α | **≤ α always** (measured 0.000–0.010) | distribution-free, finite-sample |
| Change end-to-end (v2 detector, full pipeline) | recall / FPR | **0.81 / 0.010** | detector at the ~35 px label floor |

**Best achievable on this data:** identity and surface are **perfect** (Rank-1 1.0,
EER 0; surface recall 1.0 at IOS noise, 0 % false-change). The change *measurement
and certificate* are **near-perfect**: with an accurate margin the certificate
recovers a change with recall **0.98 at a true 0 % false-progression rate** (and
**1.00** for changes ≥ 16 px), after an NCC-reliability gate on the candidate-patch
search lowered the stable-pair noise floor (this lifted the ceiling from 0.97 and
the end-to-end pipeline from 0.77). The only sub-perfect number is the
**fully-automatic** pipeline (**0.81**), capped by the tooth detector, which sits
at the DenPAR annotation-noise floor (~35 px) — not by the certificate. Its coarse
localization lands the patch off the bone margin and *attenuates* the measured
signal (a 4 px change reads ~2 px), so the smallest changes fall under the
threshold; the certificate itself is unchanged. Improving it further needs better
landmark labels than the dataset provides, not better code.

## What is genuinely solid

1. **The conformal guarantee is real and held in every ablation.** Across α ∈
   {0.05, 0.1, 0.2} and all noise levels, the false-progression / false-change
   rate stayed at or below α (change: 0.000–0.005; surface: 0.000). This is a
   distribution-free, finite-sample property — the *specificity* is trustworthy.
2. **Identification separates cleanly** on the available data: every query's
   genuine match was its nearest gallery entry (Rank-1 = 1.0, EER = 0), and 3D
   identity was **robust to tooth loss** (Rank-1 = 1.0 down to 30 % arch
   coverage) — consistent with the forensic literature.
3. **Engineering is production-grade**: the integrated library is at 100 % test
   coverage (77 tests), with a clean API and a finite-sample-correct certifier.

## The limitations that block clinical use (quantified)

1. **Validation is synthetic, not longitudinal.** There is *one* scan/radiograph
   per subject. "Genuine" identity matches are noised/partial copies of the same
   capture; "change" is a rendered geometric edit; surface change is a uniform
   displacement; the photo-reconstruction uses *rendered* views. Real
   cross-session variation (months–years apart, different devices, wear, new
   restorations, real disease) is far larger and is **not tested**. The single
   real-rescan study in the literature still had this advantage that we do not.
   → Every "1.000" should be read as an **optimistic ceiling**, not field
   performance.

2. **Change sensitivity vs noise — largely fixed by the registration measurement.**
   The *landmark*-distance scoring collapses with noise (recall 0.75 → 0.18 as
   acquisition noise goes 3 → 8 px). The production certificate instead uses the
   **differential sub-pixel registration** measurement, which holds recall
   **0.98 (3 px) → 0.96 (8 px)** before failing at extreme 15 px (0.08). So at
   realistic noise the certificate is far more sensitive than the landmark
   ablation implied; only severe repositioning (>~10 px residual) defeats it.
   *Residual:* needs confirmation on real radiograph pairs, where projection-angle
   change (not modelled here) adds error.

3. **The surface certificate needs a real scanner.** At 0.2 mm reconstruction
   noise, recall for a 1 mm change is **0**. Our own Gaussian-Splatting
   photo-reconstruction is ~0.84 mm — roughly **8× too noisy** for the surface
   certificate. The "no scanner" path and the surface certificate are therefore
   incompatible; the certificate is only usable on IOS-class scans (sub-0.1 mm).

4. **The detector is coarse** (~36 px landmark error, near the DenPAR
   annotation-noise floor). It bounds the fully-automatic pipeline: end-to-end
   change recall is **0.81** (vs the **0.98** measurement ceiling with an accurate
   margin) — the detector, not the certificate, is the ceiling.

5. **No clinical or regulatory validation whatsoever**: no prospective study, no
   radiologist/forensic-expert ground truth, no demographic diversity (age,
   ethnicity, population), no pathology (caries, implants, ortho, restorations),
   no regulatory clearance (FDA 510(k)/De Novo or CE-MDR), no ISO 13485 quality
   system. The conformal guarantee also assumes the calibration set is
   exchangeable with deployment — which requires calibrating on the *target*
   scanner and population, not synthetic noise.

## Deployment-engineering hardening (completed this round)

The technically-fixable gaps have been addressed; the system is now at
deployment-grade *engineering* (not clinical validation):

- **Noise-robust change measurement** wired through (registration, not landmarks)
  — recall 0.96–0.98 up to 8 px noise (above).
- **Site recalibration** (`clinical.SiteCalibration`) — recalibrate the conformal
  layer on the deployment site's own data, with a min-sample floor and a
  provenance hash, so the α guarantee is honoured on-distribution.
- **Input quality gates** (`clinical.assess_radiograph` / `assess_scan`) — refuse
  blurred radiographs / incomplete scans rather than certify them.
- **First-class abstention** and an **append-only audit trail**
  (`clinical.AuditLog`) for full traceability.
- **Governance docs**: [MODEL_CARD](../MODEL_CARD.md), [RISK](../RISK.md),
  [CLINICAL_READINESS](../CLINICAL_READINESS.md).
- 100 % test coverage maintained (75 tests).

What this does **not** change: there is still no real longitudinal/cross-session
data, no prospective study, and no regulatory clearance. Those are the gate.

## Verdict by use case

- **Clinical diagnostic / change-screening tool — not viable today.** Real-world
  sensitivity is unproven and low under realistic noise; there is no clinical
  validation or clearance. Using it to decide patient care would be unsafe.
- **Forensic identification aid (post-mortem ↔ ante-mortem matching) — closest to
  plausible, but still not deployable.** The method matches the forensic SOTA and
  is robust to tooth loss, but it must be validated on **real same-person,
  different-session** scans and reviewed in expert forensic studies before any
  casework use; identifications must remain expert-confirmed, never automated.

## What it would take to reach the hospital

1. **Real longitudinal & cross-session data** — same patients imaged across time
   and devices, with expert-labelled change/identity ground truth.
2. **Prospective clinical validation** with radiologist/forensic consensus,
   inter-operator and inter-device studies, on a diverse population including
   pathology.
3. **Calibrate the conformal layer on the deployment distribution**, and report
   sensitivity at the clinically-required operating point (a screening tool
   typically needs high recall, which today it lacks under noise).
4. **Regulatory pathway**: FDA/CE submission, ISO 13485 QMS, risk management
   (ISO 14971), documented failure-mode analysis.
5. **Harden the front ends**: a more precise/robust detector; a reconstruction
   path that reaches sub-0.1 mm if the surface certificate is to run without a
   scanner.

Treat the headline numbers as proof that the *methods* are sound and the
*guarantee* is real — not as evidence of clinical readiness.
