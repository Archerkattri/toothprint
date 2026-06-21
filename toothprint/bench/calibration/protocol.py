"""Calibration metadata and small-n guards."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


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


def _coverage_satisfiable(n: int, target_coverage: float) -> bool:
    """Return True iff *n* calibration points can support *target_coverage*.

    Split-conformal prediction requires
    ``ceil((n + 1) * (1 - alpha)) <= n``
    to obtain a finite interval at miscoverage level ``alpha = 1 - target_coverage``.
    Equivalently: ``n >= ceil((n+1)*(1-alpha))`` which holds when
    ``n*(1 - (1-alpha)) >= 1 - alpha``, i.e. ``n >= 1 / alpha - 1``
    (approximately). We compute it exactly.
    """
    alpha = 1.0 - target_coverage
    if alpha <= 0.0:
        return False  # perfect coverage impossible
    required_index = math.ceil((n + 1) * (1.0 - alpha)) - 1
    return required_index < n


def _n_needed(target_coverage: float) -> int:
    """Return the minimum n such that ``_coverage_satisfiable(n, target_coverage)`` is True."""
    alpha = 1.0 - target_coverage
    if alpha <= 0.0:
        return int(1e9)
    # n >= ceil(1/alpha) is the standard bound; search from there
    candidate = max(1, math.ceil(1.0 / alpha))
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
) -> "BudgetResult | dict[str, dict[str, int | bool]]":
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
        effective_coverage = (1.0 - alpha) if alpha is not None else target_coverage
        satisfiable = _coverage_satisfiable(n_cal, effective_coverage)
        needed = _n_needed(effective_coverage)
        return BudgetResult(
            n_cal=n_cal,
            alpha=alpha if alpha is not None else (1.0 - target_coverage),
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
