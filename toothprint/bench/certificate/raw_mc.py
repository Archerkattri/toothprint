"""Raw Monte-Carlo interval baseline."""

from __future__ import annotations


def raw_interval(scores: list[float] | tuple[float, ...]) -> tuple[float, float]:
    if not scores:
        raise ValueError("raw_interval requires at least one score")
    return (min(scores), max(scores))
