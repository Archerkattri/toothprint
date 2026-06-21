"""Build labeled baseline-followup pairs from perio-KPT annotations.

Since perio-KPT is a cross-sectional dataset (no true temporal follow-up),
pairs are built synthetically:

  - stable:     baseline = GT + acq_noise,  followup = GT + acq_noise
  - progressed: baseline = GT + acq_noise,  followup = GT + crestal_shift + acq_noise

Acquisition noise is drawn from a proper Gaussian distribution using
``numpy.random.default_rng`` (Box-Muller via numpy) for calibrated
residuals.  Determinism is guaranteed by a seeded RNG.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from toothprint.bench.geometry import mean_point, translate_point, translate_points
from toothprint.bench.perturb.acquisition import PerturbedPair, TransformParams


@dataclass(frozen=True)
class PairBuilderConfig:
    acq_noise_std: float = 3.0
    crestal_shift_px: float = 20.0
    seed: int = 0


# ------------------------------------------------------------------
# Legacy LCG kept for backward compatibility (imported by tests)
# ------------------------------------------------------------------

def _lcg(state: int) -> tuple[int, float]:
    """Tiny LCG returning (new_state, float in (-1, 1))."""
    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
    return state, (state / 0x80000000) - 1.0


# ------------------------------------------------------------------
# Numpy-backed Gaussian noise
# ------------------------------------------------------------------

def _noise_batch(rng: np.random.Generator, n: int, std: float) -> np.ndarray:
    """Draw *n* independent Gaussian samples with mean=0, std=*std*."""
    if std == 0.0:
        return np.zeros(n)
    return rng.normal(0.0, std, n)


def _add_noise(annotation: dict, seed: int, std: float) -> tuple[dict, int]:
    """Return a copy of annotation with each landmark jittered by Gaussian noise.

    Uses ``numpy.random.default_rng(seed)`` for reproducible Box-Muller noise.
    The second return value is ``seed + 1`` so callers that thread a running
    state through sequential calls still get different results per call.
    """
    ann = deepcopy(annotation)
    rng = np.random.default_rng(seed)

    for tooth in ann.get("teeth", []):
        for field in ("cej", "apex", "crest_line"):
            pts = tooth.get(field)
            if not pts:
                continue
            n = len(pts)
            dx_vals = _noise_batch(rng, n, std)
            dy_vals = _noise_batch(rng, n, std)
            tooth[field] = [
                translate_point(pt, dx=float(dx_vals[i]), dy=float(dy_vals[i]))
                for i, pt in enumerate(pts)
            ]

    return ann, seed + 1


def _inject_crestal_shift(annotation: dict, tooth_id: str, delta_bone: float) -> dict:
    """Return a copy with crest_line shifted by delta_bone pixels along the bone vector.

    Shifts along the CEJ→crest direction so the bone level (distance) increases
    by exactly delta_bone regardless of tooth orientation.

    Raises KeyError if tooth_id is not found (consistent with inject_crestal_change).
    """
    ann = deepcopy(annotation)
    found = False
    for tooth in ann.get("teeth", []):
        if str(tooth.get("tooth_id")) == str(tooth_id):
            found = True
            cej = tooth.get("cej")
            crest = tooth.get("crest_line")
            if not cej or not crest:
                break
            cej_mid = mean_point(cej)
            crest_mid = mean_point(crest)
            dx = crest_mid[0] - cej_mid[0]
            dy = crest_mid[1] - cej_mid[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-9:
                break
            ux, uy = dx / length, dy / length
            tooth["crest_line"] = translate_points(crest, dx=ux * delta_bone, dy=uy * delta_bone)
            break
    if not found:
        raise KeyError(f"Unknown tooth_id={tooth_id!r}")
    return ann


def build_pairs(
    records: Iterable[object],
    config: PairBuilderConfig = PairBuilderConfig(),
) -> list[PerturbedPair]:
    """Build one stable + one progressed pair per record.

    Each record must have `.annotation_dict` and `.image_id`.
    The progressed pair shifts the first available tooth's crest_line.
    """
    pairs: list[PerturbedPair] = []
    # Derive a deterministic integer seed per-record from the config seed.
    state = config.seed * 6364136223846793005 & 0xFFFFFFFF

    for record in records:
        ann = record.annotation_dict
        teeth = ann.get("teeth", [])
        if not teeth:
            continue
        # Inject the change into the first SCORABLE tooth (one with both cej and
        # crest_line). Picking teeth[0] blindly silently no-ops the shift when
        # that tooth lacks landmarks, producing a "progressed" pair with no
        # actual change — which caps recall. Skip records with no scorable tooth.
        scorable = next(
            (t for t in teeth if t.get("cej") and t.get("crest_line")), None
        )
        if scorable is None:
            continue
        first_tooth_id = str(scorable.get("tooth_id", "0"))

        # stable: both baseline and followup use GT + independent noise
        base_noisy, state = _add_noise(ann, state, config.acq_noise_std)
        state += 1
        follow_noisy, state = _add_noise(ann, state, config.acq_noise_std)
        state += 1
        pairs.append(PerturbedPair(
            baseline=base_noisy,
            followup=follow_noisy,
            label="stable",
            params=None,
            true_change=0.0,
        ))

        # progressed: followup = GT + crestal_shift + noise
        shifted = _inject_crestal_shift(ann, first_tooth_id, config.crestal_shift_px)
        base_noisy2, state = _add_noise(ann, state, config.acq_noise_std)
        state += 1
        follow_shifted_noisy, state = _add_noise(shifted, state, config.acq_noise_std)
        state += 1
        pairs.append(PerturbedPair(
            baseline=base_noisy2,
            followup=follow_shifted_noisy,
            label="progressed",
            params=None,
            true_change=config.crestal_shift_px,
        ))

    return pairs
