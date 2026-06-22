"""Modality-agnostic identification scoring (closed-set Rank-1, separability)."""

from __future__ import annotations

import numpy as np


def rank1_match(scores_row) -> int:
    """Index of the best (smallest-distance) gallery match for one query."""
    r = np.asarray(scores_row, dtype=float)
    if r.size == 0:
        raise ValueError("scores_row must be non-empty")
    return int(np.argmin(r))


def identification_metrics(scores: np.ndarray, query_labels, gallery_labels) -> dict:
    """Rank-1 accuracy + genuine/impostor separation from a distance matrix.

    ``scores[i, j]`` is the match distance (registration RMSE or constellation
    residual) of query *i* against gallery *j* — smaller means more similar.
    Returns Rank-1 accuracy, the genuine/impostor distance summaries, and the
    decidability index d' = |mu_g - mu_i| / sqrt((var_g + var_i)/2).
    """
    scores = np.asarray(scores, dtype=float)
    q = list(query_labels)
    g = list(gallery_labels)
    if scores.shape != (len(q), len(g)):
        raise ValueError("scores shape must be (n_query, n_gallery)")
    correct = 0
    genuine, impostor = [], []
    for i in range(len(q)):
        if g[rank1_match(scores[i])] == q[i]:
            correct += 1
        for j in range(len(g)):
            (genuine if g[j] == q[i] else impostor).append(scores[i, j])
    genuine = np.asarray(genuine)
    impostor = np.asarray(impostor)
    dprime = 0.0
    if genuine.size and impostor.size:
        denom = np.sqrt((genuine.var() + impostor.var()) / 2.0)
        dprime = (
            float(abs(genuine.mean() - impostor.mean()) / denom) if denom > 0 else 0.0
        )
    return {
        "rank1_accuracy": correct / len(q),
        "n_query": len(q),
        "n_gallery": len(g),
        "genuine_mean": float(genuine.mean()) if genuine.size else float("nan"),
        "genuine_max": float(genuine.max()) if genuine.size else float("nan"),
        "impostor_mean": float(impostor.mean()) if impostor.size else float("nan"),
        "impostor_min": float(impostor.min()) if impostor.size else float("nan"),
        "decidability": dprime,
    }
