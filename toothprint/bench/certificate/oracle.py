"""Oracle helpers for known perturbation upper bounds."""

from __future__ import annotations


def oracle_interval(
    score: float,
    noise_budget_px: float,
) -> tuple[float, float]:
    """Tight interval using known acquisition noise budget.

    The oracle knows the acquisition noise std; a 3-sigma bound gives
    the tightest possible certified interval.

    Parameters
    ----------
    score:
        Predicted change score (pixels).
    noise_budget_px:
        Known acquisition noise budget in pixels (e.g. 3 * noise_std).
        Defines the half-width of the certified interval.

    Returns
    -------
    tuple[float, float]
        ``(lo, hi)`` where ``lo = max(0, score - noise_budget_px)`` and
        ``hi = score + noise_budget_px``.
    """
    lo = max(0.0, score - noise_budget_px)
    hi = score + noise_budget_px
    return (round(lo, 6), round(hi, 6))
