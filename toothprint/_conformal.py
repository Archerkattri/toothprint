"""Exact finite-sample order-statistic ranks used by calibration methods."""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal


def upper_order_index(n: int, alpha: float | Decimal, *, tails: int = 1) -> int:
    """Return the zero-based upper-tail conformal order-statistic index.

    The result may equal or exceed ``n`` when the requested finite-sample
    coverage requires an unbounded interval. Decimal arithmetic preserves the
    decimal value supplied by callers at exact integer boundaries.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if tails < 1:
        raise ValueError("tails must be positive")

    miscoverage = Decimal(str(alpha)) / tails
    rank = (Decimal(n + 1) * (Decimal(1) - miscoverage)).to_integral_value(
        rounding=ROUND_CEILING
    )
    return int(rank) - 1


def lower_order_rank(n: int, alpha: float) -> int:
    """Return the one-based lower-tail conformal order-statistic rank."""
    if n < 0:
        raise ValueError("n must be non-negative")

    rank = (Decimal(n + 1) * Decimal(str(alpha))).to_integral_value(
        rounding=ROUND_FLOOR
    )
    return int(rank)
