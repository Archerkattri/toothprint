"""Naive baselines for the DMC certificate benchmark.

Three baselines:
- naive: always certify "surface stable certified" regardless of coverage or delta
- coverage_only: certify stable iff both coverage scores >= threshold, ignore delta
- uncertainty_only: certify stable iff delta_interval_mm is narrow, ignore coverage
"""
from __future__ import annotations
from dentalmapcert.certificate import CertificateOutput, CertificateInput, Label


def naive_baseline(item: CertificateInput) -> CertificateOutput:
    """Always returns 'surface stable certified'. Worst baseline."""
    return CertificateOutput(
        certificate_id=f"cert_{item.surface_region_id}_{item.capture_id_t0}_{item.capture_id_t1}",
        surface_region_id=item.surface_region_id,
        capture_id_t0=item.capture_id_t0,
        capture_id_t1=item.capture_id_t1,
        coverage_score_t0=item.coverage_score_t0,
        coverage_score_t1=item.coverage_score_t1,
        error_interval_mm_t0=item.error_interval_mm_t0,
        error_interval_mm_t1=item.error_interval_mm_t1,
        delta_interval_mm=item.delta_interval_mm,
        label="surface stable certified",
        recapture_actions=[],
    )


def coverage_only_baseline(
    item: CertificateInput,
    *,
    coverage_threshold: float = 0.75,
) -> CertificateOutput:
    """Certify stable iff both coverage scores >= threshold. Ignores delta interval."""
    if min(item.coverage_score_t0, item.coverage_score_t1) >= coverage_threshold:
        label: Label = "surface stable certified"
    else:
        label = "uncertain / recapture"

    return CertificateOutput(
        certificate_id=f"cert_{item.surface_region_id}_{item.capture_id_t0}_{item.capture_id_t1}",
        surface_region_id=item.surface_region_id,
        capture_id_t0=item.capture_id_t0,
        capture_id_t1=item.capture_id_t1,
        coverage_score_t0=item.coverage_score_t0,
        coverage_score_t1=item.coverage_score_t1,
        error_interval_mm_t0=item.error_interval_mm_t0,
        error_interval_mm_t1=item.error_interval_mm_t1,
        delta_interval_mm=item.delta_interval_mm,
        label=label,
        recapture_actions=[],
    )


def uncertainty_only_baseline(
    item: CertificateInput,
    *,
    stable_threshold_mm: float = 0.35,
) -> CertificateOutput:
    """Certify stable iff delta_interval_mm[1] <= stable_threshold_mm. Ignores coverage."""
    if item.delta_interval_mm[1] <= stable_threshold_mm:
        label: Label = "surface stable certified"
    else:
        label = "uncertain / recapture"

    return CertificateOutput(
        certificate_id=f"cert_{item.surface_region_id}_{item.capture_id_t0}_{item.capture_id_t1}",
        surface_region_id=item.surface_region_id,
        capture_id_t0=item.capture_id_t0,
        capture_id_t1=item.capture_id_t1,
        coverage_score_t0=item.coverage_score_t0,
        coverage_score_t1=item.coverage_score_t1,
        error_interval_mm_t0=item.error_interval_mm_t0,
        error_interval_mm_t1=item.error_interval_mm_t1,
        delta_interval_mm=item.delta_interval_mm,
        label=label,
        recapture_actions=[],
    )
