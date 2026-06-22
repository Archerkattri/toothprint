"""Evaluation metrics for the DMC benchmark.

All metrics operate on lists of CertificateOutput from decide_surface_change().
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from toothprint.bench.dmc.certificate import CertificateOutput


@dataclass(frozen=True)
class BenchmarkMetrics:
    n: int
    # Capture-quality metrics (Gate 0)
    capture_only_false_change_rate: (
        float  # fraction of "stable" pairs certified as changed
    )
    useful_certified_coverage: float  # fraction of regions reaching "stable certified"
    uncertain_rate: float  # fraction landing on "uncertain / recapture"
    # Conformal-quality metrics (Gate 2+)
    mean_delta_width_mm: float  # mean width of delta_interval_mm
    delta_width_std_mm: float
    # Uncertainty discrimination
    uncertainty_auc: float  # ROC-AUC of confidence (1/delta_width) vs true_labels
    # Recapture utility
    recapture_trigger_rate: float  # fraction that triggered a recapture action


def _roc_auc(scores: "np.ndarray", labels: "np.ndarray") -> float:
    """Compute ROC-AUC via the Wilcoxon-Mann-Whitney U statistic.

    AUC = P(score of positive > score of negative).
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    count = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(
        1 for p in pos for n in neg if p == n
    )
    return float(count) / (len(pos) * len(neg))


def compute_metrics(
    outputs: list[CertificateOutput],
    *,
    true_labels: list[str] | None = None,
) -> BenchmarkMetrics:
    """Compute benchmark metrics from a list of CertificateOutputs.

    true_labels: list of "stable" or "changed" ground truth (same length as outputs).
                 Required for capture_only_false_change_rate; defaults to assuming all stable.
    """
    n = len(outputs)
    if n == 0:
        return BenchmarkMetrics(
            n=0,
            capture_only_false_change_rate=0.0,
            useful_certified_coverage=0.0,
            uncertain_rate=0.0,
            mean_delta_width_mm=0.0,
            delta_width_std_mm=0.0,
            uncertainty_auc=0.0,
            recapture_trigger_rate=0.0,
        )

    _labels_provided = true_labels is not None
    if true_labels is None:
        true_labels = ["stable"] * n

    # Gate 0 metrics
    stable_indices = [i for i, lbl in enumerate(true_labels) if lbl == "stable"]
    n_stable = len(stable_indices)
    if n_stable > 0:
        false_change_count = sum(
            1 for i in stable_indices if outputs[i].label == "surface change certified"
        )
        false_change_rate = false_change_count / n_stable
    else:
        false_change_rate = 0.0

    useful_certified_coverage = (
        sum(1 for o in outputs if o.label == "surface stable certified") / n
    )

    uncertain_rate = sum(1 for o in outputs if o.label == "uncertain / recapture") / n

    # Delta-width metrics
    widths = np.array(
        [o.delta_interval_mm[1] - o.delta_interval_mm[0] for o in outputs]
    )
    mean_delta_width_mm = float(widths.mean())
    delta_width_std_mm = float(widths.std())

    # Uncertainty AUC: ROC-AUC of confidence (1/delta_width) vs true_labels
    if _labels_provided and len(true_labels) == len(outputs):
        eps = 1e-6
        widths_arr = np.array(
            [float(o.delta_interval_mm[1] - o.delta_interval_mm[0]) for o in outputs]
        )
        confidence = 1.0 / (widths_arr + eps)
        binary_labels = np.array([1 if t == "stable" else 0 for t in true_labels])
        uncertainty_auc = _roc_auc(confidence, binary_labels)
    else:
        uncertainty_auc = 0.0

    # Recapture trigger rate
    recapture_trigger_rate = sum(1 for o in outputs if len(o.recapture_actions) > 0) / n

    return BenchmarkMetrics(
        n=n,
        capture_only_false_change_rate=false_change_rate,
        useful_certified_coverage=useful_certified_coverage,
        uncertain_rate=uncertain_rate,
        mean_delta_width_mm=mean_delta_width_mm,
        delta_width_std_mm=delta_width_std_mm,
        uncertainty_auc=uncertainty_auc,
        recapture_trigger_rate=recapture_trigger_rate,
    )


def coverage_vs_false_change_curve(
    outputs: list[CertificateOutput],
    *,
    true_labels: list[str] | None = None,
    coverage_thresholds: list[float] | None = None,
) -> list[dict]:
    """Sweep coverage threshold to produce coverage vs false-change-rate curve.

    At each threshold:
    - "certified" = those with min(coverage_t0, coverage_t1) >= threshold
    - Among certified stable: false-change-rate = fraction labeled "surface change certified"
    - useful_coverage = fraction of all that are certified stable

    Returns list of {"coverage_threshold": float, "useful_coverage": float, "false_change_rate": float}
    """
    if coverage_thresholds is None:
        coverage_thresholds = list(np.linspace(0.0, 1.0, 21))

    if true_labels is None:
        true_labels = ["stable"] * len(outputs)

    n_total = len(outputs)
    results = []

    for threshold in coverage_thresholds:
        # Filter to outputs with sufficient coverage at both timepoints
        certified_indices = [
            i
            for i, o in enumerate(outputs)
            if min(o.coverage_score_t0, o.coverage_score_t1) >= threshold
        ]

        # Among certified, find those that are truly stable
        certified_stable_indices = [
            i for i in certified_indices if true_labels[i] == "stable"
        ]
        n_certified_stable = len(certified_stable_indices)

        # False-change-rate: among certified stable, fraction labeled "surface change certified"
        if n_certified_stable > 0:
            false_change_count = sum(
                1
                for i in certified_stable_indices
                if outputs[i].label == "surface change certified"
            )
            false_change_rate = false_change_count / n_certified_stable
        else:
            false_change_rate = 0.0

        # Useful coverage: fraction of all outputs that are certified AND labeled stable
        n_certified_and_stable_label = sum(
            1
            for i in certified_indices
            if outputs[i].label == "surface stable certified"
        )
        useful_coverage = n_certified_and_stable_label / n_total if n_total > 0 else 0.0

        results.append(
            {
                "coverage_threshold": float(threshold),
                "useful_coverage": useful_coverage,
                "false_change_rate": false_change_rate,
            }
        )

    return results
