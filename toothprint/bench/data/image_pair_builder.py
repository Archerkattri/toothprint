"""Build complete image + annotation pairs for evaluation.

Since there is no trained detector, "predicted" landmarks come from GT
annotations with simulated detection noise (via pair_builder.py).  This
module additionally loads and perturbs the actual image pixels so callers
can visualise and save results.

Image arrays are uint8 HxWx3 numpy arrays.  Pillow and numpy are hard
dependencies; their absence raises ImportError at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image as _PILImage

from toothprint.bench.data.pair_builder import PairBuilderConfig, build_pairs
from toothprint.bench.perturb.image_perturb import (
    ImagePerturbConfig,
    apply_image_perturbation,
    random_image_perturb_config,
)


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageAnnotationPair:
    """A single baseline/followup pair with both images and annotations."""

    baseline_image: Any  # numpy uint8 HxWx3
    followup_image: Any  # numpy uint8 HxWx3
    baseline_annotation: dict
    followup_annotation: dict
    label: str  # "stable" | "progressed"
    image_id: str


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_image_pairs(
    records: Iterable,
    config: PairBuilderConfig = PairBuilderConfig(),  # noqa: B008
    image_perturb_seed: int = 0,
) -> list[ImageAnnotationPair]:
    """Build one stable + one progressed ImageAnnotationPair per record.

    Parameters
    ----------
    records:
        Iterable of PerioKptRecord instances.  Each must expose
        ``.image_path`` (Path), ``.annotation_dict`` (dict), and
        ``.image_id`` (str).
    config:
        PairBuilderConfig controlling annotation-level perturbations.
    image_perturb_seed:
        Base seed for deterministic image perturbation configs.
        Baseline uses ``image_perturb_seed + pair_index * 2``;
        followup uses ``image_perturb_seed + pair_index * 2 + 1``.

    Returns
    -------
    list[ImageAnnotationPair]
        Two pairs per record (stable then progressed).
    """
    records_list = list(records)

    # Build annotation-level pairs (2 per record: stable + progressed)
    ann_pairs = build_pairs(records_list, config)

    # Index records by position so we can match them to their annotation pairs.
    # build_pairs yields 2 pairs per record in order, so pair[2*i] and
    # pair[2*i+1] correspond to records_list[i].
    result: list[ImageAnnotationPair] = []
    pair_index = 0
    rec_index = 0

    # We need to skip records without teeth (build_pairs skips them too).
    # Replicate the same skipping logic here.
    valid_records = [r for r in records_list if r.annotation_dict.get("teeth")]

    for rec in valid_records:
        if pair_index >= len(ann_pairs):
            break

        raw_image = _load_image(rec.image_path)

        # stable pair
        stable_pair = ann_pairs[pair_index]
        base_cfg = random_image_perturb_config(image_perturb_seed + rec_index * 4)
        follow_cfg = random_image_perturb_config(image_perturb_seed + rec_index * 4 + 1)
        result.append(
            ImageAnnotationPair(
                baseline_image=_perturb_to_uint8(raw_image, base_cfg),
                followup_image=_perturb_to_uint8(raw_image, follow_cfg),
                baseline_annotation=stable_pair.baseline,
                followup_annotation=stable_pair.followup,
                label=stable_pair.label,
                image_id=rec.image_id,
            )
        )
        pair_index += 1

        if pair_index >= len(ann_pairs):
            break

        # progressed pair
        progressed_pair = ann_pairs[pair_index]
        base_cfg2 = random_image_perturb_config(image_perturb_seed + rec_index * 4 + 2)
        follow_cfg2 = random_image_perturb_config(
            image_perturb_seed + rec_index * 4 + 3
        )
        result.append(
            ImageAnnotationPair(
                baseline_image=_perturb_to_uint8(raw_image, base_cfg2),
                followup_image=_perturb_to_uint8(raw_image, follow_cfg2),
                baseline_annotation=progressed_pair.baseline,
                followup_annotation=progressed_pair.followup,
                label=progressed_pair.label,
                image_id=rec.image_id,
            )
        )
        pair_index += 1
        rec_index += 1

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_image(image_path: Path) -> Any:
    """Load an image as a float32 HxWx3 numpy array in [0, 1] using PIL."""
    image_path = Path(image_path)
    img = _PILImage.open(image_path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def _float_to_uint8(arr: Any) -> Any:
    """Convert a float [0,1] numpy array to uint8 [0,255]."""
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _perturb_to_uint8(image: Any, cfg: ImagePerturbConfig) -> Any:
    """Apply image perturbation and return a uint8 array."""
    perturbed = apply_image_perturbation(image, cfg)
    return _float_to_uint8(perturbed)
