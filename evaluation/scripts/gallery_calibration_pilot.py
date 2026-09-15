"""Synthetic pilot for ToothPrint's gallery-level selection calibration.

The pilot deliberately compares a pairwise lower-tail threshold with the
complete-gallery minimum-distance threshold. It demonstrates the selection
effect only; it is not longitudinal, clinical, forensic or external benchmark
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from toothprint.identity.gallery import (  # noqa: E402
    GalleryCalibration,
    evaluate_gallery_calibration,
)


def _source_digest() -> str:
    h = hashlib.sha256()
    for path in (Path(__file__), ROOT / "toothprint" / "identity" / "gallery.py"):
        h.update(str(path.relative_to(ROOT)).encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def _impostor_matrix(rng: np.random.Generator, n: int, gallery_size: int) -> np.ndarray:
    # Lower is a more convincing match. The lower tail is intentionally broad
    # enough that searching 40 candidates materially changes the event rate.
    return np.clip(rng.lognormal(mean=0.25, sigma=0.65, size=(n, gallery_size)), 0.01, 20.0)


def run(seed: int = 23, n_calibration: int = 240, n_evaluation: int = 240, gallery_size: int = 40) -> dict:
    rng = np.random.default_rng(seed)
    cal_imp = _impostor_matrix(rng, n_calibration, gallery_size)
    eval_imp = _impostor_matrix(rng, n_evaluation, gallery_size)
    calibration = GalleryCalibration.fit(
        cal_imp,
        alpha=0.05,
        site_id="synthetic-site",
        created_utc="2026-09-14T00:00:00Z",
        min_calibration=100,
    )
    heldout = evaluate_gallery_calibration(
        calibration,
        impostor_distances=eval_imp,
        gallery_labels=[f"patient-{i}" for i in range(gallery_size)],
    )

    # A pairwise threshold calibrates individual comparisons, then reuses it
    # after a 40-way search. That is the failure mode the new primitive avoids.
    flattened = np.sort(cal_imp.ravel(), kind="mergesort")
    pair_rank = int(np.floor((flattened.size + 1) * calibration.alpha))
    pair_threshold = float(flattened[pair_rank - 1]) if pair_rank > 0 else -np.inf
    pairwise_rate = float(np.mean(eval_imp.min(axis=1) < pair_threshold))

    return {
        "format": "TOOTHPRINT_GALLERY_CALIBRATION_PILOT",
        "protocol": "synthetic_non_enrolled_complete_gallery_selection",
        "seed": seed,
        "gallery_size": gallery_size,
        "n_calibration": n_calibration,
        "n_evaluation": n_evaluation,
        "source_digest": _source_digest(),
        "gallery_level": {
            "calibration": calibration.to_dict(include_scores=False),
            "heldout": heldout,
        },
        "pairwise_reuse_control": {
            "threshold_distance": None if pair_threshold == -np.inf else pair_threshold,
            "heldout_false_accept_rate_after_gallery_search": pairwise_rate,
            "claim_boundary": "control only; pairwise threshold is not a gallery-level certificate",
        },
        "claim_boundary": [
            "synthetic CPU protocol only",
            "no real patient/session data, clinical validation, forensic validation or external baseline",
            "gallery-level false-accept control is conditional on the declared gallery, selector and population",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results" / "gallery_calibration_pilot.json")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    payload = run(seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "source_digest": payload["source_digest"],
        "gallery_threshold": payload["gallery_level"]["calibration"]["threshold_distance"],
        "gallery_false_accept_rate": payload["gallery_level"]["heldout"]["false_accept_rate"],
        "pairwise_reuse_false_accept_rate": payload["pairwise_reuse_control"]["heldout_false_accept_rate_after_gallery_search"],
    }, indent=2))


if __name__ == "__main__":
    main()
