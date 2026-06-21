"""Pixel-level acquisition perturbations for dental radiograph images.

Operates on HxWx3 numpy arrays of normalized floats in [0, 1].

Perturbation types
------------------
brightness_delta : additive brightness shift in [-0.3, 0.3]
contrast_scale   : multiplicative contrast around the mean, in [0.8, 1.2]
noise_std        : standard deviation of additive Gaussian noise in [0, 0.05]
flip_horizontal  : mirror the image left-right
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

# Type alias: a 3D numpy array (HxWx3 float).
ImageArray = np.ndarray


@dataclass(frozen=True)
class ImagePerturbConfig:
    brightness_delta: float = 0.0   # additive; range [-0.3, 0.3]
    contrast_scale: float = 1.0     # multiplicative around mean; range [0.8, 1.2]
    noise_std: float = 0.0          # Gaussian noise std; range [0, 0.05]
    flip_horizontal: bool = False


def random_image_perturb_config(seed: int) -> ImagePerturbConfig:
    """Return a deterministic ImagePerturbConfig derived from *seed*."""
    rng = random.Random(seed)
    return ImagePerturbConfig(
        brightness_delta=rng.uniform(-0.3, 0.3),
        contrast_scale=rng.uniform(0.8, 1.2),
        noise_std=rng.uniform(0.0, 0.05),
        flip_horizontal=rng.random() < 0.5,
    )


def apply_image_perturbation(
    image_array: ImageArray,
    config: ImagePerturbConfig,
) -> ImageArray:
    """Apply *config* perturbations to *image_array* and return the result.

    Parameters
    ----------
    image_array:
        HxWx3 normalized float ``numpy.ndarray`` (values in [0, 1]).
    config:
        Perturbation configuration.

    Returns
    -------
    ``numpy.ndarray`` with values clamped to [0, 1].
    """
    return _apply_numpy(image_array, config)


# ------------------------------------------------------------------
# numpy implementation
# ------------------------------------------------------------------

def _apply_numpy(arr: "np.ndarray", config: ImagePerturbConfig) -> "np.ndarray":
    arr = arr.astype(float)

    if config.flip_horizontal:
        arr = arr[:, ::-1, :]

    if config.contrast_scale != 1.0:
        mean = arr.mean()
        arr = (arr - mean) * config.contrast_scale + mean

    if config.brightness_delta != 0.0:
        arr = arr + config.brightness_delta

    if config.noise_std > 0.0:
        rng = np.random.default_rng(seed=_derive_noise_seed(config))
        arr = arr + rng.normal(0.0, config.noise_std, arr.shape)

    arr = np.clip(arr, 0.0, 1.0)
    return arr


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _derive_noise_seed(config: ImagePerturbConfig) -> int:
    """Return a reproducible integer seed from the config parameters."""
    # Combine the float parameters into a stable integer seed.
    # Uses a simple but deterministic hash over the string repr.
    key = (
        f"{config.brightness_delta:.8f}"
        f"{config.contrast_scale:.8f}"
        f"{config.noise_std:.8f}"
        f"{int(config.flip_horizontal)}"
    )
    h = 0
    for ch in key:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h
