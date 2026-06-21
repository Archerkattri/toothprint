"""Deterministic threshold baseline."""

from __future__ import annotations

import math


def decide(score: float, tau: float) -> str:
    if not math.isfinite(score):
        raise ValueError("score must be finite")
    return "progressed" if score > tau else "stable"
