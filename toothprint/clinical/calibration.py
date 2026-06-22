"""Site recalibration of the conformal layer.

A conformal guarantee only holds when the calibration data is exchangeable with
deployment. A model calibrated on one scanner / population does NOT carry its
false-alarm guarantee to another. Before clinical use, the conformal layer must
be **recalibrated on the deploying site's own no-change pairs**; this module fits
that calibration, versions it, and records the provenance needed for audit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from toothprint.change.conformal import ConformalCertifier


def data_fingerprint(values) -> str:
    """SHA-256 of a measurement set — provenance for which data calibrated a model."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


@dataclass(frozen=True)
class SiteCalibration:
    """A conformal certifier fitted on one site's data, with provenance."""

    certifier: ConformalCertifier
    site_id: str
    n_calibration: int
    alpha: float
    data_sha256: str
    created_utc: str

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
        if measured.size < min_calibration:
            raise ValueError(
                f"site calibration needs >= {min_calibration} stable pairs, got {measured.size}"
            )
        cert = ConformalCertifier.fit(
            measured, np.asarray(true_stable, dtype=np.float64), alpha=alpha
        )
        return cls(
            certifier=cert,
            site_id=site_id,
            n_calibration=int(measured.size),
            alpha=alpha,
            data_sha256=data_fingerprint(measured),
            created_utc=created_utc,
        )

    @property
    def calibration_id(self) -> str:
        """Stable identifier for this calibration (site + data hash + time)."""
        return f"{self.site_id}:{self.data_sha256[:12]}:{self.created_utc}"

    def to_dict(self) -> dict:
        return {
            "site_id": self.site_id,
            "n_calibration": self.n_calibration,
            "alpha": self.alpha,
            "data_sha256": self.data_sha256,
            "created_utc": self.created_utc,
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
        )
