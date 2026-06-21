"""Detector-based change evaluation on real image pairs (measurable recall).

Unlike the annotation-only synthetic protocol — where baseline and follow-up
share identical pixels so a real detector sees zero change — this builds genuine
image pairs:

  * stable   : the follow-up is a re-acquisition of the same geometry (sensor
               noise only), so the true bone-level change is 0.
  * progressed: the follow-up has the crest **rendered moved** in pixels by
               ``render_crestal_change`` (true change = ``delta_px``).

ViTPose is run independently on the baseline and follow-up images, and the
change score is the difference of the two predicted CEJ-to-crest distances.
Because both timepoints are the same tooth in nearly the same crop, the
detector's common-mode bias cancels in the difference — the noise floor is the
detector's *repeatability*, not its ~38 px absolute error — which is what makes
recall measurable at clinically meaningful change magnitudes.
"""
from __future__ import annotations

import numpy as np

from toothprint.bench.geometry import mean_point
from toothprint.bench.landmarks.vitpose_detector import predict_tooth
from toothprint.bench.perturb.image_change import render_crestal_change
from toothprint.bench.score.periodontal import scalar_change_score


def scorable_teeth(record) -> list:
    """Teeth that carry both cej and crest_line (needed for a bone-level score)."""
    return [
        t for t in record.annotation_dict.get("teeth", [])
        if t.get("cej") and t.get("crest_line")
    ]


def acquire(image: np.ndarray, rng: np.random.Generator, noise_std: float) -> np.ndarray:
    """Simulate a follow-up acquisition: additive sensor noise, no geometry change."""
    if noise_std <= 0.0:
        return image.copy()
    noise = rng.normal(0.0, noise_std, image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(image.dtype)


def evaluate_real_image_pairs(
    records,
    detector,
    *,
    delta_px: float,
    acq_noise_std: float = 3.0,
    max_teeth_per_image: int = 3,
    seed: int = 0,
    image_loader=None,
) -> list[dict]:
    """Return per-tooth (stable + progressed) rows with detector change scores.

    Each row: ``{label, score, true_change, image_id, tooth_id}`` where ``score``
    is the detector's measured CEJ-to-crest change (follow-up minus baseline) and
    ``true_change`` is 0.0 (stable) or ``delta_px`` (progressed).

    ``detector`` exposes ``predict_crop``; ``image_loader(path) -> HxWx3 uint8``
    defaults to PIL.
    """
    if image_loader is None:
        from PIL import Image as _Image

        def image_loader(path):  # noqa: E306
            return np.array(_Image.open(path).convert("RGB"))

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for rec in records:
        img = image_loader(rec.image_path)
        for tooth in scorable_teeth(rec)[:max_teeth_per_image]:
            base_pred = predict_tooth(detector, img, tooth)
            if base_pred is None:
                continue

            stable_img = acquire(img, rng, acq_noise_std)
            stable_pred = predict_tooth(detector, stable_img, tooth)

            cej_mid = mean_point(tooth["cej"])
            crest_mid = mean_point(tooth["crest_line"])
            prog_img = render_crestal_change(img, cej_mid, crest_mid, delta_px)
            prog_img = acquire(prog_img, rng, acq_noise_std)
            prog_pred = predict_tooth(detector, prog_img, tooth)

            if stable_pred is None or prog_pred is None:
                continue

            base_ann = {"teeth": [base_pred]}
            tid = base_pred["tooth_id"]
            try:
                stable_score = scalar_change_score(base_ann, {"teeth": [stable_pred]}, tooth_id=tid)
                prog_score = scalar_change_score(base_ann, {"teeth": [prog_pred]}, tooth_id=tid)
            except (KeyError, ValueError):
                continue

            rows.append({"label": "stable", "score": stable_score, "true_change": 0.0,
                         "image_id": rec.image_id, "tooth_id": tid})
            rows.append({"label": "progressed", "score": prog_score, "true_change": float(delta_px),
                         "image_id": rec.image_id, "tooth_id": tid})
    return rows


def summarize_real_image_decisions(rows: list[dict], conformal, tau: float) -> dict:
    """Classify each row's conformal interval vs ``tau`` and aggregate metrics."""
    from toothprint.bench.certificate.conformal import classify_interval

    n_stable = n_prog = 0
    fp = tp = uncertain = stable_cert = 0
    for r in rows:
        decision = classify_interval(conformal.predict(r["score"]), tau=tau)
        if r["label"] == "stable":
            n_stable += 1
            if decision == "progressed":
                fp += 1
            elif decision == "stable":
                stable_cert += 1
            else:
                uncertain += 1
        else:
            n_prog += 1
            if decision == "progressed":
                tp += 1
            elif decision != "stable":
                uncertain += 1
    return {
        "n_stable": n_stable,
        "n_progressed": n_prog,
        "true_change_recall": (tp / n_prog) if n_prog else 0.0,
        "false_progression_rate": (fp / n_stable) if n_stable else 0.0,
        "stable_certification_rate": (stable_cert / n_stable) if n_stable else 0.0,
    }
