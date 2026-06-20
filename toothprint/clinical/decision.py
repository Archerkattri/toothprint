"""Clinical decision wrapper: quality gate -> certify -> abstain -> audit.

Ties the deployment pieces together into one auditable call. Abstention is
first-class: if the input fails the quality gate the system returns
``refer / recapture`` rather than a change verdict, and every outcome — including
abstention — is written to the audit log with full provenance.
"""
from __future__ import annotations

from dataclasses import dataclass

from toothprint.clinical.audit import AuditLog, AuditRecord
from toothprint.clinical.calibration import SiteCalibration
from toothprint.clinical.quality import QualityReport

REFER = "refer / recapture"


@dataclass(frozen=True)
class ClinicalDecision:
    label: str
    measured: float
    interval: tuple
    quality_ok: bool
    provenance: dict


def clinical_certify(measured: float, *, site_cal: SiteCalibration, tau: float,
                     quality: QualityReport, input_fingerprint: str, operator: str,
                     timestamp_utc: str, audit_log: "AuditLog | None" = None) -> ClinicalDecision:
    """Certify a measured change under a site calibration, gating on input quality.

    Returns ``refer / recapture`` (abstain) when the input quality gate fails,
    otherwise the conformal verdict (changed / stable / uncertain). The outcome is
    appended to ``audit_log`` if provided.
    """
    if not quality.usable:
        label, interval = REFER, (float("nan"), float("nan"))
    else:
        interval = site_cal.certifier.interval(measured)
        label = site_cal.certifier.classify(measured, tau)
    provenance = {"calibration_id": site_cal.calibration_id,
                  "input_fingerprint": input_fingerprint,
                  "timestamp_utc": timestamp_utc, "operator": operator,
                  "quality_issues": list(quality.issues)}
    if audit_log is not None:
        audit_log.record(AuditRecord(
            timestamp_utc=timestamp_utc, input_fingerprint=input_fingerprint,
            calibration_id=site_cal.calibration_id, decision=label,
            measured=float(measured), interval=tuple(interval), operator=operator))
    return ClinicalDecision(label=label, measured=float(measured), interval=tuple(interval),
                            quality_ok=quality.usable, provenance=provenance)
