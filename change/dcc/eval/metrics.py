"""Canonical DentalChangeCert outcome metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

LABELS = ("stable", "progressed")
DECISIONS = ("stable", "progressed", "uncertain")


@dataclass(frozen=True)
class DecisionSummary:
    table: dict[str, dict[str, int]]
    false_progression_rate: float
    true_change_recall: float
    stable_certification_rate: float
    progression_certification_rate: float
    uncertain_rate: float
    n: int
    mean_interval_width: float
    interval_width_std: float

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_decisions(rows: list[dict[str, str]]) -> DecisionSummary:
    table = {label: {decision: 0 for decision in DECISIONS} for label in LABELS}
    for row in rows:
        true = row["true"]
        decision = row["decision"]
        if true not in table:
            raise ValueError(f"Unknown true label: {true}")
        if decision not in table[true]:
            raise ValueError(f"Unknown decision: {decision}")
        table[true][decision] += 1

    stable_total = sum(table["stable"].values())
    progressed_total = sum(table["progressed"].values())
    n = stable_total + progressed_total
    false_progression_rate = _safe_div(table["stable"]["progressed"], stable_total)
    true_change_recall = _safe_div(table["progressed"]["progressed"], progressed_total)
    stable_certification_rate = _safe_div(table["stable"]["stable"], stable_total)
    progression_certification_rate = _safe_div(table["progressed"]["progressed"], progressed_total)
    uncertain_rate = _safe_div(table["stable"]["uncertain"] + table["progressed"]["uncertain"], n)

    widths = [row["hi"] - row["lo"] for row in rows if "hi" in row and "lo" in row]
    # Abstaining (unbounded) intervals have inf width; degenerate inputs can
    # produce NaN. Exclude non-finite widths so a single abstaining certificate
    # does not poison the aggregate mean/std with inf/NaN.
    finite_widths = np.asarray(widths, dtype=float)
    finite_widths = finite_widths[np.isfinite(finite_widths)]
    if finite_widths.size:
        mean_interval_width = float(finite_widths.mean())
        interval_width_std = float(finite_widths.std(ddof=0))
    else:
        mean_interval_width = 0.0
        interval_width_std = 0.0

    return DecisionSummary(
        table=table,
        false_progression_rate=false_progression_rate,
        true_change_recall=true_change_recall,
        stable_certification_rate=stable_certification_rate,
        progression_certification_rate=progression_certification_rate,
        uncertain_rate=uncertain_rate,
        n=n,
        mean_interval_width=mean_interval_width,
        interval_width_std=interval_width_std,
    )


def coverage_vs_false_progression_curve(
    rows: list[dict],
    tau: float,
    n_points: int = 20,
) -> list[dict]:
    """Sweep interval width factor to produce coverage vs false-progression curve.

    At each width factor w:
    - Scaled interval: [center - w*(half_width), center + w*(half_width)]
    - A stable pair has "false progression" if its scaled interval lower bound > tau
    - A stable pair is "certified stable" if its scaled interval upper bound < tau

    Returns list of {"width_factor": float, "false_prog_rate": float, "stable_cert_rate": float}
    sorted by width_factor ascending.
    """
    width_factors = np.linspace(0.1, 3.0, n_points).tolist()
    curve = []
    for w in width_factors:
        fp = 0
        cert = 0
        n_stable = 0
        for row in rows:
            center = (row["lo"] + row["hi"]) / 2.0
            half = (row["hi"] - row["lo"]) / 2.0
            lo_scaled = center - w * half
            hi_scaled = center + w * half
            is_stable = row.get("true") == "stable"
            if is_stable:
                n_stable += 1
                if lo_scaled > tau:
                    fp += 1
                if hi_scaled < tau:
                    cert += 1
        fpr = fp / n_stable if n_stable > 0 else 0.0
        csr = cert / n_stable if n_stable > 0 else 0.0
        curve.append({"width_factor": float(w), "false_prog_rate": float(fpr), "stable_cert_rate": float(csr)})
    return curve


def tightness_at_fixed_coverage(
    rows: list[dict],
    tau: float,
    target_coverage: float = 0.90,
    n_points: int = 200,
) -> float | None:
    """Return the smallest width factor achieving target stable-cert coverage.

    Sweeps width factors on a finer grid than coverage_vs_false_progression_curve.
    Returns None if even width_factor=3.0 doesn't reach target_coverage.
    """
    for w in np.linspace(0.01, 3.0, n_points):
        cert = 0
        n_stable = 0
        for row in rows:
            center = (row["lo"] + row["hi"]) / 2.0
            half = (row["hi"] - row["lo"]) / 2.0
            hi_scaled = center + w * half
            if row.get("true") == "stable":
                n_stable += 1
                if hi_scaled < tau:
                    cert += 1
        if n_stable > 0 and cert / n_stable >= target_coverage:
            return float(w)
    return None


def auc_fpr_coverage_curve(curve: list[dict]) -> float:
    """Trapezoidal AUC of FPR vs stable-cert-rate curve.

    Args:
        curve: output of coverage_vs_false_progression_curve()

    Returns:
        Area under the curve in [0, 1]. Lower = better (less false-progression
        at any coverage level). Returns NaN when the curve has fewer than two
        points or a degenerate (zero-width) x-axis, since the AUC is undefined
        there — returning 0.0 (the "best" value) would falsely signal a perfect
        certificate on an empty/collapsed evaluation.
    """
    if len(curve) < 2:
        return float("nan")
    x = np.array([pt["stable_cert_rate"] for pt in curve])
    y = np.array([pt["false_prog_rate"] for pt in curve])
    if float(np.ptp(x)) == 0.0:
        return float("nan")
    # Sort by x (stable_cert_rate) ascending for trapezoid integration.
    # np.trapz was removed in NumPy 2.0 and renamed np.trapezoid. Select by
    # hasattr — NOT getattr(np, "trapezoid", np.trapz), whose default argument
    # eagerly evaluates np.trapz and raises AttributeError on NumPy 2.x.
    _trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    order = np.argsort(x)
    return float(_trapezoid(y[order], x[order]))


def _safe_div(num: int, den: int) -> float:
    return 0.0 if den == 0 else round(num / den, 10)


@dataclass(frozen=True)
class DifficultySlice:
    name: str
    summary: DecisionSummary


def slice_by_difficulty(
    rows: list[dict],
    *,
    noise_low_threshold: float = 3.0,
    noise_high_threshold: float = 6.0,
    shift_threshold: float = 10.0,
) -> list[DifficultySlice]:
    """Partition rows into difficulty strata and summarize each.

    Strata (each row appears in exactly one):
    - "stable_low_noise": true=="stable", score < noise_low_threshold
    - "stable_medium_noise": true=="stable", noise_low_threshold <= score < noise_high_threshold
    - "stable_high_noise": true=="stable", score >= noise_high_threshold
    - "progressed_small_shift": true=="progressed", score < shift_threshold
    - "progressed_large_shift": true=="progressed", score >= shift_threshold

    Any row missing "score" goes into its own stratum by label only.
    """
    strata: dict[str, list] = {
        "stable_low_noise": [],
        "stable_medium_noise": [],
        "stable_high_noise": [],
        "progressed_small_shift": [],
        "progressed_large_shift": [],
    }
    for row in rows:
        score = row.get("score", row.get("predicted_score", 0.0))
        label = row.get("true", "stable")
        if label == "stable":
            if score < noise_low_threshold:
                strata["stable_low_noise"].append(row)
            elif score < noise_high_threshold:
                strata["stable_medium_noise"].append(row)
            else:
                strata["stable_high_noise"].append(row)
        else:
            if score < shift_threshold:
                strata["progressed_small_shift"].append(row)
            else:
                strata["progressed_large_shift"].append(row)

    slices = []
    for name, slice_rows in strata.items():
        if not slice_rows:
            continue
        slices.append(DifficultySlice(name=name, summary=summarize_decisions(slice_rows)))
    return slices
