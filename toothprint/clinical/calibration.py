"""Site recalibration of the conformal layer.

A conformal guarantee only holds when the calibration data is exchangeable with
deployment. A model calibrated on one scanner / population does NOT carry its
false-alarm guarantee to another. Before clinical use, the conformal layer must
be **recalibrated on the deploying site's own no-change pairs**; this module fits
that calibration, versions it, and records the provenance needed for audit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from toothprint.change.conformal import ConformalCertifier


CALIBRATION_SPEC_VERSION = 2
CALIBRATION_METHOD_VERSION = "site-conformal-cqr-v2"


def data_fingerprint(values) -> str:
    """SHA-256 of a canonical numeric array, without storing its values."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    h = hashlib.sha256()
    h.update(b"toothprint-array-v2\0")
    h.update(json.dumps({"dtype": arr.dtype.str, "shape": list(arr.shape)},
                        separators=(",", ":"), sort_keys=True).encode("ascii"))
    h.update(b"\0")
    h.update(arr.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class SiteCalibration:
    """A conformal certifier fitted on one site's data, with provenance."""

    certifier: ConformalCertifier
    site_id: str
    n_calibration: int
    alpha: float
    data_sha256: str
    created_utc: str
    true_sha256: str = ""
    minimum_calibration: int = 100
    method_version: str = CALIBRATION_METHOD_VERSION
    spec_version: int = CALIBRATION_SPEC_VERSION

    @classmethod
    def fit(
        cls,
        measured_stable,
        true_stable,
        *,
        site_id: str,
        created_utc: str,
        alpha: float = 0.1,
        min_calibration: int = 100,
    ) -> "SiteCalibration":
        """Calibrate on the site's stable (no-change) pairs.

        Raises if too few calibration points to support the requested guarantee —
        the finite-sample conformal bound needs n >= ~1/alpha; ``min_calibration``
        enforces a clinically defensible floor rather than silently under-covering.
        """
        measured = np.asarray(measured_stable, dtype=np.float64)
        truth = np.asarray(true_stable, dtype=np.float64)
        if measured.shape != truth.shape:
            raise ValueError("measured_stable and true_stable must have the same shape")
        if not np.isfinite(measured).all() or not np.isfinite(truth).all():
            raise ValueError("calibration arrays must be finite")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if min_calibration < 1:
            raise ValueError("min_calibration must be >= 1")
        if measured.size < min_calibration:
            raise ValueError(
                f"site calibration needs >= {min_calibration} stable pairs, got {measured.size}"
            )
        cert = ConformalCertifier.fit(
            measured, truth, alpha=alpha
        )
        return cls(
            certifier=cert,
            site_id=site_id,
            n_calibration=int(measured.size),
            alpha=alpha,
            data_sha256=data_fingerprint(measured),
            created_utc=created_utc,
            true_sha256=data_fingerprint(truth),
            minimum_calibration=int(min_calibration),
        )

    @property
    def calibration_id(self) -> str:
        """Stable hash of the complete versioned calibration specification."""
        spec = {
            "spec_version": self.spec_version,
            "method_version": self.method_version,
            "site_id": self.site_id,
            "created_utc": self.created_utc,
            "n_calibration": self.n_calibration,
            "minimum_calibration": self.minimum_calibration,
            "alpha": self.alpha,
            "measured_sha256": self.data_sha256,
            "true_sha256": self.true_sha256,
            "q_lo": self.certifier.q_lo,
            "q_hi": self.certifier.q_hi,
        }
        encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"{self.site_id}:{hashlib.sha256(encoded).hexdigest()[:16]}:{self.created_utc}"

    def to_dict(self) -> dict:
        return {
            "site_id": self.site_id,
            "n_calibration": self.n_calibration,
            "alpha": self.alpha,
            "data_sha256": self.data_sha256,
            "true_sha256": self.true_sha256,
            "created_utc": self.created_utc,
            "minimum_calibration": self.minimum_calibration,
            "method_version": self.method_version,
            "spec_version": self.spec_version,
            "q_lo": self.certifier.q_lo,
            "q_hi": self.certifier.q_hi,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SiteCalibration":
        return cls(
            certifier=ConformalCertifier(
                q_lo=d["q_lo"], q_hi=d["q_hi"], alpha=d["alpha"]
            ),
            site_id=d["site_id"],
            n_calibration=d["n_calibration"],
            alpha=d["alpha"],
            data_sha256=d["data_sha256"],
            created_utc=d["created_utc"],
            true_sha256=d.get("true_sha256", ""),
            minimum_calibration=d.get("minimum_calibration", d.get("n_calibration", 0)),
            method_version=d.get("method_version", "site-conformal-cqr-v1"),
            spec_version=d.get("spec_version", 1),
        )
