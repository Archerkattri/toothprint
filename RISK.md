# ToothPrint — Risk Analysis

A hazard analysis in the spirit of ISO 14971 (medical-device risk management).
This is an engineering risk register, **not** a completed regulatory risk file —
that requires a quality system and clinical evidence (see CLINICAL_READINESS.md).

Severity: Minor / Serious / Critical. Each hazard lists the implemented
mitigation and the residual risk that real-world validation must still address.

| # | Hazard | Cause | Severity | Mitigation (implemented) | Residual risk |
|---|---|---|---|---|---|
| H1 | **False "no change"** — real disease progression missed | detector-limited end-to-end sensitivity (0.81); unproven on real longitudinal data | Critical | conformal abstention ("uncertain"); registration measurement (noise- **and repositioning**-robust, multi-anchor affine); surface regional detection of localized lesions; report sensitivity at the operating point | High until validated on real longitudinal data; **never use as sole screen** |
| H2 | **False "change"** — unnecessary intervention | reconstruction/measurement noise read as change | Serious | conformal false-alarm rate bounded by α; verified ≤ α in all ablations | Bound holds only if calibrated on the deployment distribution (H6) |
| H3 | **Mis-identification** — wrong person matched | impostor closer than genuine; out-of-gallery query | Critical | smallest-RMSE + separation reported; **closed-set only**; expert confirmation mandated | Open-set / cross-session not validated; must show candidates, never auto-decide |
| H4 | **Garbage-in verdict** — confident output on unusable capture | blurred radiograph / incomplete scan | Serious | `clinical.assess_radiograph` / `assess_scan` quality gates → abstain ("refer / recapture") | Thresholds need per-site tuning on real captures |
| H5 | **Untraceable decision** — no record for audit/forensics | missing provenance | Serious | append-only `clinical.AuditLog` (input hash, calibration id, decision, operator, time) | Must be wired into the deploying system's record-keeping |
| H6 | **Guarantee voided by distribution shift** — α bound fails off-distribution | calibrated on a different scanner/population | Critical | `clinical.SiteCalibration` recalibration + provenance hash + min-sample floor | Requires the site to actually recalibrate and re-validate |
| H7 | **Silent failure** — degraded output instead of an error | fallbacks masking real problems | Serious | no fallbacks anywhere: bad input / missing dependency / unmeasurable patch **raise** | — |
| H8 | **Biased accuracy** — worse for some groups | non-diverse validation data | Serious | documented in MODEL_CARD as out-of-scope until validated | High — needs diverse prospective data |
| H9 | **Privacy breach** — biometric gallery exposed | enrolment data is identifying | Serious | library stores no patient data; integrator owns consent/encryption/retention | Deployment-dependent |

## Risk control principles built in

- **Abstain over guess.** Every mechanism can return "uncertain / recapture".
- **No fallbacks.** Failures surface as errors, not degraded results.
- **Provenance on everything.** Calibration and inputs are hashed and logged.
- **Closed-set, expert-in-the-loop.** Identification proposes; a human decides.

The hazards above with "High" residual risk are the reasons ToothPrint is not yet
deployable; closing them is gated on real clinical data and a regulatory process,
not on further code.
