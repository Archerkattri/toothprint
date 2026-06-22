"""High-level radiograph bone-level change certificate.

Ties the differential registration measurement to the conformal certifier: given
two timepoint images and the tooth's CEJ/crest, measure the apical bone-level
change and return a certified label with its conformal interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toothprint.change.conformal import ConformalCertifier
from toothprint.change.registration import measure_change, measure_change_search


def bone_vector(cej_mid, crest_mid) -> np.ndarray:
    """Unit apical (CEJ -> crest) direction; raises if the points coincide."""
    v = np.asarray(crest_mid, dtype=float) - np.asarray(cej_mid, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("CEJ and crest coincide; bone vector undefined")
    return v / n


@dataclass(frozen=True)
class ChangeCertificate:
    measured_px: float
    interval_px: tuple
    label: str
    response: float


def certify_change(
    g0: np.ndarray,
    g1: np.ndarray,
    reference_center,
    crest_center,
    bone_unit,
    certifier: ConformalCertifier,
    tau: float,
    offsets=None,
) -> ChangeCertificate:
    """Measure and certify the bone-level change between two timepoint images.

    ``offsets`` enables the candidate-patch search for coarse localisation; pass
    ``None`` for a single precise patch. Raises if the bone-margin patch cannot be
    measured (out of bounds).
    """
    if offsets is None:
        out = measure_change(g0, g1, reference_center, crest_center, bone_unit)
    else:
        out = measure_change_search(
            g0, g1, reference_center, crest_center, bone_unit, offsets
        )
    if out is None:
        raise ValueError("could not measure the bone-margin patch (out of bounds)")
    measured, response = out
    return ChangeCertificate(
        measured_px=measured,
        interval_px=certifier.interval(measured),
        label=certifier.classify(measured, tau),
        response=response,
    )
