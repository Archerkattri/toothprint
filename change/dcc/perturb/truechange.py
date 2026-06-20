"""Controlled true periodontal change edits."""

from __future__ import annotations

import math
from copy import deepcopy

from dcc.geometry import mean_point, translate_points
from dcc.perturb.acquisition import PerturbedPair


def inject_crestal_change(annotation: dict, tooth_id: str, delta_px: float) -> PerturbedPair:
    """Inject a controlled crestal bone-loss change by shifting crest_line along the bone vector.

    The shift is applied along the unit vector from the CEJ midpoint to the crest
    midpoint, so the measured bone-level distance increases by exactly *delta_px*
    regardless of tooth orientation.

    Parameters
    ----------
    annotation:
        Source annotation dict (not mutated).
    tooth_id:
        Which tooth to perturb.
    delta_px:
        Magnitude of the crestal shift in pixels (positive = bone loss).

    Returns
    -------
    PerturbedPair
        Baseline is a deep copy of the original annotation; followup has the
        shifted crest_line for the specified tooth.
    """
    followup = deepcopy(annotation)
    found = False
    for tooth in followup.get("teeth", []):
        if str(tooth.get("tooth_id")) == str(tooth_id):
            cej = tooth.get("cej", [])
            crest = tooth.get("crest_line", [])
            if not (cej and crest):
                raise ValueError(
                    f"tooth_id={tooth_id} needs both cej and crest_line to inject a "
                    "bone-vector crestal change; no fallback is provided."
                )
            cej_mid = mean_point(cej)
            crest_mid = mean_point(crest)
            dx = crest_mid[0] - cej_mid[0]
            dy = crest_mid[1] - cej_mid[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-9:
                raise ValueError(
                    f"tooth_id={tooth_id} CEJ and crest coincide; bone vector undefined."
                )
            ux, uy = dx / length, dy / length
            tooth["crest_line"] = translate_points(crest, dx=ux * delta_px, dy=uy * delta_px)
            found = True
            break
    if not found:
        raise KeyError(f"Unknown tooth_id={tooth_id}")
    return PerturbedPair(
        baseline=deepcopy(annotation),
        followup=followup,
        label="progressed",
        params=None,
        true_change=delta_px,
    )


def is_local_crest_change(baseline: dict, followup: dict, tooth_id: str) -> bool:
    base_tooth = _find_tooth(baseline, tooth_id)
    follow_tooth = _find_tooth(followup, tooth_id)
    anchors_same = (
        base_tooth.get("cej") == follow_tooth.get("cej")
        and base_tooth.get("apex") == follow_tooth.get("apex")
    )
    crest_changed = base_tooth.get("crest_line") != follow_tooth.get("crest_line")
    return anchors_same and crest_changed


def _find_tooth(annotation: dict, tooth_id: str) -> dict:
    for tooth in annotation.get("teeth", []):
        if str(tooth.get("tooth_id")) == str(tooth_id):
            return tooth
    raise KeyError(f"Unknown tooth_id={tooth_id}")
