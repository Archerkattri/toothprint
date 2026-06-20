# ToothPrint — thorough evaluation & hospital-readiness assessment

A full, ablated evaluation of all three mechanisms on all available real data
(Poseidon3D intraoral scans, DenPAR radiographs). Raw results are in
`evaluation/results/*.json`; the degradation curves are in
`evaluation/ablation_summary.png`. Reproduce with the `scripts/eval_*.py`
harness in the companion repositories.

> **Bottom line up front.** The *methods and code* are now hardened to
> medical-grade *engineering*: the conformal false-alarm guarantee holds in every
> ablation (specificity is unconditional), the change measurement is robust to
> acquisition noise **and** repositioning (rotation/magnification, not just
> translation), the surface certificate is de-biased and detects **localized**
> lesions a whole-surface average misses, and identity separates cleanly at
> **N=400** (2D) / **N=80** (3D). **But this is still NOT a deployable medical
> device, and not "perfect":** every headline number is measured against
> **synthetic perturbations of single-timepoint data** (no real
> longitudinal/cross-session dental data exists in these datasets), and there is no
> prospective study and no regulatory clearance. Those gaps are **non-code** — they
> cannot be closed by any amount of engineering — and they remain the gate. Details
> below.

## Headline metrics (real data + synthetic perturbations)

| Mechanism | Metric | Result | Caveat |
|---|---|---|---|
| Identity — 3D scans (**N=80**) | Rank-1 / EER / AUC | **1.000 / 0.000 / 1.000** | genuine = synthetic re-scan; Rank-1 1.0 holds under non-rigid deformation ≤0.2 mm |
| Identity — 2D radiographs (**N=400**, both splits, ≥4 teeth) | Rank-1 / EER / AUC | **1.000 / 0.000 / 1.000** | robust to 20 px jitter (0.985) & 50 % magnification; partly by-design invariance |
| Surface certificate — global change (de-biased) | recall @1 mm (σ≤0.4 mm) / FPR | **1.000 / 0.000** | usable recon-noise **0.1 → 0.4 mm** (de-biasing, was 0.1); 0.84 mm photo-recon still too noisy |
| Surface certificate — **localized** change (regional) | recall @1 mm (σ=0.2 mm) / FPR | **0.99 / 0.000** | global average gets **0.00** (dilutes); regional max localizes it; FPR ≤ α via max-calibration |
| Change **measurement** (accurate localization) | recall @ **true 0 %** false-progression | **0.98** (0.95 at 4 px → **1.00** at ≥16 px) | tau = 2 px; FPR = 0.000 |
| Change certificate | false-progression rate vs α | **≤ α always** (measured 0.000–0.010) | distribution-free, finite-sample |
| Change end-to-end (v2 detector, full pipeline) | recall / FPR | **0.81 / 0.010** | detector at the ~35 px label floor |

**Best achievable on this data:** identity and surface are **perfect** in their
usable regime (Rank-1 1.0, EER 0; surface recall 1.0 with 0 % false-change, now to
**0.4 mm** reconstruction noise after de-biasing, was 0.1 mm). The change *measurement
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
   coverage (97 tests), with a clean API and a finite-sample-correct certifier.

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

2. **Change sensitivity vs noise — fixed by the registration measurement; and now
   robust to *repositioning*, not just translation.** The *landmark*-distance
   scoring collapses with noise (recall 0.75 → 0.18 as acquisition noise goes
   3 → 8 px); the production certificate instead uses the **differential sub-pixel
   registration** measurement, which holds recall **0.98 (3 px) → 0.96 (8 px)**.
   The earlier residual — a single crown reference cancels only a global
   *translation*, so rotation and projection **magnification** between visits leak
   straight into the measurement — is closed by a **multi-anchor affine
   global-motion model** (`fit_global_motion` + `measure_change_anchored`): it fits
   the motion from several stationary crown anchors and cancels it *at the crest*.
   On real DenPAR teeth with **no** real change, the stable-pair spurious change
   under repositioning drops ~8×: at a mild 1°/2 %/4 px, single-reference already
   leaks **5.4 px** (a false ~0.5 mm progression) vs anchored **0.6 px**; at a
   severe 4°/8 %/16 px, **22.4 px → 2.9 px**
   (`docs/repositioning_robustness_v2.png`). *Residual:* the perturbations are
   synthetic — real longitudinal pairs (with true projection-geometry change and
   tissue change) remain the validation gate.

3. **The surface certificate: de-biased *and* localized.** The original mean-norm
   measurement *rectifies* zero-mean reconstruction noise into a +2.3·σ bias, so its
   conformal radius grew linearly with noise (0.069 → 0.458 mm) and recall for a
   1 mm change collapsed by 0.2 mm — and a whole-surface average can't see a
   *localized* lesion at all (the signal is diluted by the unchanged majority).
   Two fixes, both shipped with their ablations:
   - **De-biasing** (noise-power subtraction, `surface_displacement`:
     √(max(0, mean‖v‖² − floor))) cuts the radius ~6× and extends the usable
     reconstruction noise from 0.1 → **0.4 mm** (recall 1.0 @ 1 mm; IOS detection
     threshold 1.0 → 0.8 mm).
   - **Regional detection** (`assign_regions` + `regional_displacements`: de-biased
     displacement per region, max over K=12, conformal-calibrated on the *max* so
     false-change stays ≤ α despite the multiplicity) detects a localized patch
     change that the global average **dilutes to 0** — global recall **0.00 →
     regional 0.99** @ 1 mm (σ=0.2 mm) — and reports *which* region moved.
   **Honest residuals:** (a) the photo-reconstruction — now a high-detail
   **2DGS+TSDF mesh at 0.42 mm median** (`make_gsplat_mesh.py`, up from the 0.84 mm
   point cloud) — sits right at this certificate's 0.4 mm usable-noise edge: its
   *median* clears the bar but its *mean* (~0.6 mm) and tail do not, so a 1 mm
   *global* change on the raw 0.84 mm point cloud still reads recall 0.02 and an IOS
   scan remains preferable; (b) under
   heavy correlated noise a *small* (1 mm) localized change is hard (regional recall
   0.04 at correlation 0.9), though a 1.5 mm one is recoverable (**0.48**, vs
   global's 0.00). **Specificity (0 % false-change) holds at every noise level,
   correlation, and region count** — unconditional; the *sensitivity* reach on tiny
   changes under coherent error must be pinned down on real reconstruction residuals.

4. **The detector is coarse** (~36 px landmark error, near the DenPAR
   annotation-noise floor). It bounds the fully-automatic pipeline: end-to-end
   change recall is **0.81** (vs the **0.98** measurement ceiling with an accurate
   margin) — the detector, not the certificate, is the ceiling. We **rigorously
   confirmed** this is irreducible *in code*: edge-snapping the crest onto the bone
   margin (`snap_to_margin`) and widening the synthetic warp both give **no lift**
   (end-to-end recall stays 0.79–0.82 across both), because the candidate-patch
   search already extracts the available signal at a misplaced patch. Closing it
   needs a better detector (better landmark labels), not better measurement code.

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
- **Repositioning-robust change** — a multi-anchor affine global-motion model
  cancels rotation + magnification, not just translation (~8× lower stable-pair
  spurious change on real teeth; caveat 2).
- **De-biased + regional surface certificate** — noise-power subtraction extends
  the usable reconstruction noise to 0.4 mm, and a max-over-regions statistic
  detects *localized* lesions a whole-surface average misses (0.99 vs 0.00), with
  the conformal false-change rate preserved (caveat 3).
- **Larger, harder identity validation** — 2D gallery grown to **N=400** at
  Rank-1 1.0 / EER 0 (robust to 20 px jitter, 50 % magnification); 3D identity
  (N=80) holds **Rank-1 1.0 under non-rigid intra-subject deformation up to
  0.2 mm** (wear / soft-tissue movement, on top of pose + noise + partial coverage)
  and **0.93 at a large 0.4 mm** — measured, not asserted (caveat 1, tightened).
- **Dentist-usable mesh reconstruction** — the "no scanner" path now produces a
  watertight **1.2 M-triangle mesh** via Gaussian-Splatting + multi-view **TSDF
  fusion** (`make_gsplat_mesh.py`), matching the GT scan to **0.42 mm median** (was
  an 0.84 mm point cloud) — importable into dental CAD, not just a point cloud.
- **Evidence videos** — animated identification (genuine-hugs / impostor-floats)
  and reconstruction (mesh + error heatmap) turntables, plus a change-measurement-
  in-action figure, replace the static/weak figures in the README.
- **Site recalibration** (`clinical.SiteCalibration`) — recalibrate the conformal
  layer on the deployment site's own data, with a min-sample floor and a
  provenance hash, so the α guarantee is honoured on-distribution.
- **Input quality gates** (`clinical.assess_radiograph` / `assess_scan`) — refuse
  blurred radiographs / incomplete scans rather than certify them.
- **First-class abstention** and an **append-only audit trail**
  (`clinical.AuditLog`) for full traceability.
- **Governance docs**: [MODEL_CARD](../MODEL_CARD.md), [RISK](../RISK.md),
  [CLINICAL_READINESS](../CLINICAL_READINESS.md).
- 100 % test coverage maintained (97 tests).

What this does **not** change: there is still no real longitudinal/cross-session
data, no prospective study, and no regulatory clearance. Those are the gate.

## Verdict by use case

- **Clinical diagnostic / change-screening tool — not viable today.** The
  measurement is now robust to acquisition noise *and* repositioning, but real-world
  sensitivity is **unproven on real longitudinal data** (synthetic robustness is not
  field validation), and there is no clinical validation or clearance. Using it to
  decide patient care would be unsafe.
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
   sensitivity at the clinically-required operating point (a screening tool needs
   high recall there — the *measurement* reaches it, but the detector-limited
   end-to-end pipeline (0.81) and the absence of real-data validation do not yet).
4. **Regulatory pathway**: FDA/CE submission, ISO 13485 QMS, risk management
   (ISO 14971), documented failure-mode analysis.
5. **Harden the front ends**: a more precise/robust detector; a reconstruction
   path that reaches ~0.4 mm (the de-biased surface certificate's usable noise) —
   or better, if it is to run without a scanner. Pin the surface reach down on
   *real* reconstruction residuals, since the de-biasing gain assumes the noise is
   spatially incoherent.

Treat the headline numbers as proof that the *methods* are sound and the
*guarantee* is real — not as evidence of clinical readiness.
