"""Fusion-ready summaries shared with DentalChangeCert."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from toothprint.bench.dmc.calibration import ErrorCalibrator
from toothprint.bench.dmc.certificate import CertificateOutput


@dataclass(frozen=True)
class FusionTimepoint:
    subject_id: str
    timepoint_id: str
    dental_map_capture_id: str
    dental_change_radiograph_id: str | None
    visible_surface_certificates: list[CertificateOutput]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["visible_surface_certificates"] = [cert.to_dict() for cert in self.visible_surface_certificates]
        return payload

    @property
    def has_radiograph_pair(self) -> bool:
        return self.dental_change_radiograph_id is not None


def to_fusion_timepoint(
    coverage_dict: dict[str, float],
    calibrator: ErrorCalibrator,
    subject_id: str = "unknown",
    timepoint_id: str = "t0",
    capture_id: str = "cap_0",
    radiograph_id: str | None = None,
) -> FusionTimepoint:
    """Build a :class:`FusionTimepoint` from a coverage dict and a fitted calibrator.

    This is the standard factory for creating fusion summaries from the output
    of :func:`~dentalmapcert.capture_protocol.coverage_per_region`.  One
    :class:`~dentalmapcert.certificate.CertificateOutput` is synthesised for
    each region in *coverage_dict*.

    Args:
        coverage_dict:  Mapping of region_id -> coverage fraction in [0, 1].
            Typically from :func:`~dentalmapcert.capture_protocol.coverage_per_region`.
        calibrator:     A fitted :class:`~dentalmapcert.calibration.ErrorCalibrator`.
        subject_id:     Subject identifier.
        timepoint_id:   Timepoint label (e.g. ``"t0"``).
        capture_id:     Dental-map capture identifier.
        radiograph_id:  Optional paired radiograph identifier.

    Returns:
        A :class:`FusionTimepoint` ready for serialisation or downstream fusion.
    """
    from toothprint.bench.dmc.certificate import CertificateInput, decide_surface_change

    certs: list[CertificateOutput] = []
    err_interval = calibrator.interval(0.0)
    for region, cov in coverage_dict.items():
        inp = CertificateInput(
            surface_region_id=f"{subject_id}_{region}",
            capture_id_t0=capture_id,
            capture_id_t1=capture_id,
            coverage_score_t0=cov,
            coverage_score_t1=cov,
            error_interval_mm_t0=err_interval,
            error_interval_mm_t1=err_interval,
            delta_interval_mm=(0.0, calibrator.radius_mm),
            region_type=region if region in (
                "anterior_crown",
                "buccal_crown",
                "lingual_or_palatal_crown",
                "occlusal_or_incisal",
                "visible_gingival_margin",
            ) else "unknown_visible_dental_surface",
        )
        certs.append(decide_surface_change(inp))

    return FusionTimepoint(
        subject_id=subject_id,
        timepoint_id=timepoint_id,
        dental_map_capture_id=capture_id,
        dental_change_radiograph_id=radiograph_id,
        visible_surface_certificates=certs,
    )

