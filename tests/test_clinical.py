import json

import numpy as np
import pytest

from toothprint.clinical.audit import AuditLog, AuditRecord, input_fingerprint
from toothprint.clinical.calibration import SiteCalibration, data_fingerprint
from toothprint.clinical.decision import REFER, ClinicalDecision, clinical_certify
from toothprint.clinical.quality import QualityReport, assess_radiograph, assess_scan

TS = "2026-06-21T00:00:00Z"


# --- calibration -----------------------------------------------------------


def _stable(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1.0, n), np.zeros(n)


def test_site_calibration_fit_and_id():
    m, t = _stable()
    sc = SiteCalibration.fit(m, t, site_id="hospital_A", created_utc=TS, alpha=0.1)
    assert sc.n_calibration == 120 and sc.alpha == 0.1
    assert sc.certifier.q_lo > 0 and sc.certifier.q_hi > 0
    assert sc.site_id in sc.calibration_id and sc.created_utc in sc.calibration_id


def test_site_calibration_too_few_raises():
    with pytest.raises(ValueError, match=">= 100 stable pairs"):
        SiteCalibration.fit(np.zeros(20), np.zeros(20), site_id="x", created_utc=TS)


def test_site_calibration_roundtrip():
    m, t = _stable()
    sc = SiteCalibration.fit(m, t, site_id="A", created_utc=TS)
    sc2 = SiteCalibration.from_dict(sc.to_dict())
    assert sc2.certifier.q_lo == sc.certifier.q_lo and sc2.site_id == "A"
    assert sc2.calibration_id == sc.calibration_id


def test_data_fingerprint_stable_and_sensitive():
    assert data_fingerprint([1, 2, 3]) == data_fingerprint([1, 2, 3])
    assert data_fingerprint([1, 2, 3]) != data_fingerprint([1, 2, 4])


# --- quality ---------------------------------------------------------------


def test_assess_radiograph_usable():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (256, 256)).astype(float)  # sharp, high contrast
    r = assess_radiograph(img)
    assert r.usable and r.issues == [] and r.metrics["width"] == 256


def test_assess_radiograph_blurred_and_lowcontrast():
    flat = np.full((256, 256), 100.0)  # no sharpness, no contrast
    r = assess_radiograph(flat)
    assert not r.usable
    assert any("blurred" in i for i in r.issues) and any(
        "contrast" in i for i in r.issues
    )


def test_assess_radiograph_too_small():
    rng = np.random.default_rng(1)
    r = assess_radiograph(rng.integers(0, 255, (64, 64)).astype(float))
    assert not r.usable and any("too small" in i for i in r.issues)


def test_assess_radiograph_bad_ndim():
    with pytest.raises(ValueError, match="2D greyscale"):
        assess_radiograph(np.zeros((4, 4, 3)))


def test_assess_scan_usable_and_sparse():
    rng = np.random.default_rng(2)
    good = rng.normal(0, 15, (3000, 3))
    assert assess_scan(good).usable
    bad = rng.normal(0, 15, (500, 3))
    rep = assess_scan(bad)
    assert not rep.usable and any("sparse" in i for i in rep.issues)


def test_assess_scan_too_small_extent():
    pts = np.random.default_rng(3).normal(0, 1.0, (3000, 3))  # tiny extent (~10mm)
    rep = assess_scan(pts)
    assert not rep.usable and any("incomplete arch" in i for i in rep.issues)


def test_assess_scan_empty():
    rep = assess_scan(np.zeros((0, 3)))
    assert not rep.usable and rep.metrics["extent_mm"] == 0.0


def test_assess_scan_bad_shape():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        assess_scan(np.zeros((10, 2)))


# --- audit -----------------------------------------------------------------


def test_input_fingerprint():
    a = np.arange(10.0)
    assert input_fingerprint(a) == input_fingerprint(a.copy())
    assert input_fingerprint(a) != input_fingerprint(a + 1)


def test_audit_log_record_and_export(tmp_path):
    log = AuditLog()
    rec = AuditRecord(TS, "fp", "calA", "changed", 1.2, (0.9, 1.5), "drX")
    log.record(rec)
    assert len(log) == 1 and list(log)[0].decision == "changed"
    p = log.to_jsonl(tmp_path / "audit.jsonl")
    line = json.loads(p.read_text().strip())
    assert line["decision"] == "changed" and line["interval"] == [0.9, 1.5]


# --- decision --------------------------------------------------------------


def _cal():
    m, t = _stable()
    return SiteCalibration.fit(m, t, site_id="A", created_utc=TS, alpha=0.1)


def test_clinical_certify_changed_and_audits():
    log = AuditLog()
    q = QualityReport(usable=True, metrics={}, issues=[])
    d = clinical_certify(
        50.0,
        site_cal=_cal(),
        tau=2.0,
        quality=q,
        input_fingerprint="fp",
        operator="drX",
        timestamp_utc=TS,
        audit_log=log,
    )
    assert isinstance(d, ClinicalDecision) and d.label == "changed" and d.quality_ok
    assert len(log) == 1 and d.provenance["operator"] == "drX"


def test_clinical_certify_stable_no_log():
    q = QualityReport(usable=True, metrics={}, issues=[])
    d = clinical_certify(
        0.0,
        site_cal=_cal(),
        tau=2.0,
        quality=q,
        input_fingerprint="fp",
        operator="drX",
        timestamp_utc=TS,
    )
    assert d.label == "stable"


def test_clinical_certify_abstains_on_bad_quality():
    log = AuditLog()
    q = QualityReport(usable=False, metrics={}, issues=["too blurred"])
    d = clinical_certify(
        50.0,
        site_cal=_cal(),
        tau=2.0,
        quality=q,
        input_fingerprint="fp",
        operator="drX",
        timestamp_utc=TS,
        audit_log=log,
    )
    assert d.label == REFER and not d.quality_ok
    assert np.isnan(d.interval[0]) and len(log) == 1
    assert "too blurred" in d.provenance["quality_issues"]
