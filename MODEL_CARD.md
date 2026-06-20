# ToothPrint — Model Card

A model card for the three ToothPrint mechanisms, in the spirit of Mitchell et
al. (2019), adapted for a medical/forensic context. Read alongside
[CLINICAL_READINESS.md](CLINICAL_READINESS.md) and [RISK.md](RISK.md).

## Intended use

- **Research and method development** in dental biometrics and certified change
  detection.
- **Decision-support, expert-in-the-loop**, *after* site-specific validation:
  - forensic dental identification (a ranked candidate list for an expert to
    confirm — never an automated identity decision);
  - longitudinal radiograph / surface change flagging for clinician review.

The system outputs a **certificate with an explicit abstention** ("uncertain /
recapture"). It is designed to defer, not to over-claim.

## Out-of-scope / prohibited use

- **Autonomous clinical or forensic decisions.** Every output must be confirmed
  by a qualified professional.
- **Any patient-affecting use without** site recalibration, prospective clinical
  validation, and the applicable regulatory clearance (FDA / CE-MDR).
- Populations, devices, or pathologies outside the validated distribution.

## Models and data

| Mechanism | Method | Validation data |
|---|---|---|
| Identity (3D) | FPFH → RANSAC → ICP, smallest-RMSE | Poseidon3D intraoral scans (single session/subject) |
| Identity (2D) | landmark constellation, scale-norm rigid ICP | DenPAR radiographs (single timepoint) |
| Change | sub-pixel differential registration + conformal | DenPAR + synthetic rendered change |
| Surface | scale-aware ICP / Poisson error + conformal | Poseidon3D + synthetic displacement |

**Front ends** (ViTPose tooth detection, Gaussian-Splatting reconstruction) are
pluggable and optional; the certified logic runs without a GPU.

## Performance (on validation data + synthetic perturbations)

See [evaluation/REPORT.md](evaluation/REPORT.md) and
[evaluation/ablation_summary.png](evaluation/ablation_summary.png). Headlines:
identification Rank-1 = 1.000 / EER = 0 (synthetic re-scans); conformal
false-alarm rate ≤ α in every ablation; change recall and surface recall are
strong only in their good-quality regimes and degrade under noise (quantified).

## Limitations (critical — see REPORT.md for the numbers)

1. Validated on **synthetic perturbations of single-timepoint data**; no real
   longitudinal or cross-session data. Headline metrics are optimistic ceilings.
2. Change sensitivity **drops sharply with acquisition noise**.
3. Surface certificate **requires IOS-class reconstruction** (sub-0.1 mm);
   photo reconstruction is too noisy for it.
4. Tooth detection is **coarse**; end-to-end change recall ≈ 0.72.
5. **No** demographic, device, or pathology diversity in validation.

## Ethical considerations

- **Dental biometrics is sensitive personal data.** Enrolment galleries are
  identifying and must be consented, access-controlled, encrypted, and
  retention-limited; ToothPrint stores none of this — it is a library.
- **Bias risk**: performance is unverified across age, ethnicity, and dentition
  states; deploying without diverse validation risks unequal accuracy.
- **Forensic gravity**: a false identification has severe consequences;
  identifications must stay expert-confirmed with the candidate evidence shown.

## Maintenance

Recalibrate the conformal layer per deployment site/device
(`toothprint.clinical.SiteCalibration`) and re-validate when the imaging pipeline
changes. All certificates should be logged (`toothprint.clinical.AuditLog`).
