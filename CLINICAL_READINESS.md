# ToothPrint — Clinical Readiness

An honest, line-by-line status of what stands between ToothPrint and lawful,
safe hospital use. **Status: NOT clinically deployable.** The engineering is
hardened to deployment grade; the gate is real-world validation and regulatory
clearance — neither of which can be produced from code or synthetic data.

Legend:  ✅ done  ·  🟡 partial / engineering-only  ·  ⬜ requires real data / an
institution / a regulator (cannot be done in software).

## 1. Algorithmic & engineering quality

| Item | Status |
|---|---|
| Sound, current methods (registration, conformal, FPFH/ICP, Gaussian Splatting) | ✅ |
| Finite-sample false-alarm guarantee, verified ≤ α across ablations | ✅ |
| No fallbacks — failures raise, not degrade | ✅ |
| First-class abstention ("uncertain / recapture") | ✅ |
| Input quality gates (refuse unusable captures) | ✅ |
| Site recalibration + provenance hashing | ✅ |
| Append-only audit trail | ✅ |
| 100% unit-test coverage (94 tests) | ✅ |
| Change measurement robust to acquisition noise (registration vs landmarks) | 🟡 improved; needs real-pair confirmation |
| Photo reconstruction accurate enough for the surface certificate | 🟡 0.84 mm; de-biasing lifted the bar to ~0.4 mm but 0.84 mm is still too noisy (needs IOS scanner or larger change) |
| Detector localisation precision | 🟡 ~36 px; coarse |

## 2. Clinical validation — none of this is software

| Item | Status |
|---|---|
| Real **multi-session / longitudinal** data (same patients over time) | ⬜ |
| Real cross-device, cross-operator captures | ⬜ |
| Expert (radiologist / forensic odontologist) ground truth | ⬜ |
| Prospective clinical study with a pre-registered protocol | ⬜ |
| Demographic diversity (age, sex, ethnicity, edentulous/pathology) | ⬜ |
| Pathology robustness (caries, restorations, implants, ortho) | ⬜ |
| Sensitivity reported at the clinically-required operating point | ⬜ |
| Open-set identification validation (query may be absent from gallery) | ⬜ |

## 3. Regulatory & quality system — none of this is software

| Item | Status |
|---|---|
| Intended-use / indications-for-use statement | 🟡 drafted in MODEL_CARD |
| ISO 14971 risk management file | 🟡 register started in RISK.md |
| ISO 13485 quality management system | ⬜ |
| FDA 510(k)/De Novo or CE-MDR conformity assessment | ⬜ |
| Clinical evaluation report | ⬜ |
| Cybersecurity & data-protection (HIPAA/GDPR) controls in the deployed system | ⬜ |
| Post-market surveillance plan | ⬜ |

## The path (in order)

1. Partner with a clinical/forensic institution; obtain IRB approval and consent.
2. Acquire **real longitudinal & cross-session** data with expert labels.
3. Recalibrate (`clinical.SiteCalibration`) and **re-measure** all metrics on real
   data — expect lower numbers than the synthetic ceilings.
4. Run a prospective, pre-registered clinical study on a diverse population.
5. Stand up an ISO 13485 QMS; compile the ISO 14971 file and clinical evaluation.
6. Pursue FDA/CE clearance for the specific indication.
7. Only then: supervised, expert-in-the-loop clinical pilot.

**No step from #2 onward can be completed in code.** Until they are, ToothPrint
is a validated *research prototype*, used only for research and method development.
