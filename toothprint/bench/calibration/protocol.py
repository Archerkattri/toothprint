"""Calibration metadata and small-n guards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING, Decimal

from toothprint._conformal import upper_order_index


@dataclass(frozen=True)
class CalibrationRecord:
    stratum: str
    predicted_score: float
    observed_score: float
    source: str

    @property
    def absolute_residual(self) -> float:
        return abs(float(self.predicted_score) - float(self.observed_score))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["absolute_residual"] = self.absolute_residual
        return payload


@dataclass(frozen=True)
class CalibrationBudget:
    min_per_stratum: int = 30
    target_coverage: float = 0.9


@dataclass(frozen=True)
class BudgetResult:
    """Result of a simple calibration budget check (non-stratum form)."""

    n_cal: int
    alpha: float
    target_coverage: float
    coverage_satisfiable: bool
    n_needed: int


def _coverage_satisfiable(n: int, target_coverage: float | Decimal) -> bool:
    """Return True iff *n* calibration points can support *target_coverage*.

    Split-conformal prediction requires
    ``ceil((n + 1) * (1 - alpha)) <= n``
    to obtain a finite interval at miscoverage level ``alpha = 1 - target_coverage``.
    A finite order statistic exists exactly when ``(n + 1) * alpha >= 1``,
    where ``alpha = 1 - target_coverage``.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    exact_coverage = Decimal(str(target_coverage))
    exact_alpha = Decimal(1) - exact_coverage
    if not Decimal(0) < exact_coverage < Decimal(1):
        return False  # perfect coverage impossible
    required_index = upper_order_index(n, exact_alpha)
    return 0 <= required_index < n


def _n_needed(target_coverage: float) -> int:
    """Return the minimum n such that ``_coverage_satisfiable(n, target_coverage)`` is True."""
    exact_coverage = Decimal(str(target_coverage))
    if not Decimal(0) < exact_coverage < Decimal(1):
        return int(1e9)
    alpha = Decimal(1) - exact_coverage
    # Finite radius exists exactly when (n + 1) * alpha >= 1.
    reciprocal_ceiling = (Decimal(1) / alpha).to_integral_value(
        rounding=ROUND_CEILING
    )
    candidate = max(1, int(reciprocal_ceiling) - 1)
    while not _coverage_satisfiable(candidate, target_coverage):  # pragma: no cover
        candidate += 1
        if candidate > 100_000:
            return candidate
    return candidate


def check_calibration_budget(
    n_cal: int | None = None,
    alpha: float | None = None,
    target_coverage: float = 0.9,
    records: list[CalibrationRecord] | None = None,
    budget: CalibrationBudget = CalibrationBudget(),
) -> BudgetResult | dict[str, dict[str, int | bool]]:
    """Check whether the calibration set is large enough.

    Two calling conventions are supported:

    **Simple form** (used by run_gate2.py)::

        result = check_calibration_budget(n_cal=100, alpha=0.1, target_coverage=0.9)
        # Returns a BudgetResult with .coverage_satisfiable and .n_needed

    **Stratum form** (legacy, used by tests)::

        result = check_calibration_budget(records=my_records, budget=CalibrationBudget())
        # Returns a dict keyed by stratum name

    Parameters
    ----------
    n_cal:
        Total number of calibration samples (simple form).
    alpha:
        Miscoverage level 1 - target_coverage (simple form).  If provided,
        ``target_coverage`` is derived as ``1 - alpha``.
    target_coverage:
        Desired marginal coverage (default 0.9).
    records:
        List of CalibrationRecord objects (stratum form, legacy).
    budget:
        CalibrationBudget controlling minimum-per-stratum threshold (stratum form).
    """
    # Simple form: n_cal + alpha kwargs
    if n_cal is not None:
        if n_cal < 0:
            raise ValueError("n_cal must be non-negative")
        if alpha is not None:
            if not 0.0 < alpha < 1.0:
                raise ValueError("alpha must be in (0, 1)")
            exact_coverage = Decimal(1) - Decimal(str(alpha))
        else:
            exact_coverage = Decimal(str(target_coverage))
        effective_coverage = float(exact_coverage)
        satisfiable = _coverage_satisfiable(n_cal, exact_coverage)
        needed = _n_needed(exact_coverage)
        return BudgetResult(
            n_cal=n_cal,
            alpha=float(Decimal(1) - exact_coverage),
            target_coverage=effective_coverage,
            coverage_satisfiable=satisfiable,
            n_needed=needed,
        )

    # Stratum form (legacy): records list
    if records is None:
        records = []
    counts: dict[str, int] = {}
    for record in records:
        counts[record.stratum] = counts.get(record.stratum, 0) + 1
    return {
        stratum: {
            "n": n,
            "min_required": budget.min_per_stratum,
            "ok": n >= budget.min_per_stratum,
            "coverage_satisfiable": _coverage_satisfiable(n, budget.target_coverage),
        }
        for stratum, n in sorted(counts.items())
    }
