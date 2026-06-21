"""Image quality detection utilities for DentalMapCert.

Uses OpenCV to detect blur and glare in smartphone dental images.
These scores feed into coverage_per_region to auto-populate quality tags
without requiring manual annotation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Laplacian variance below this threshold → blur (calibrated on 1080p dental images)
_BLUR_THRESHOLD = 80.0
# Maximum plausible Laplacian variance for normalisation
_BLUR_MAX_VAR = 2000.0

# Fraction of HSV value-channel pixels above this level considered "highlight"
_GLARE_VALUE_THRESHOLD = 240
# If this fraction of pixels are near-white highlights → glare
_GLARE_PIXEL_FRACTION = 0.02


def detect_blur(image: np.ndarray) -> float:
    """Return blur score in [0, 1] where 1 = sharp, 0 = very blurry.

    Uses the variance of the Laplacian of the greyscale image.  The raw
    variance is clamped to [0, _BLUR_MAX_VAR] then normalised to [0, 1].

    Args:
        image: BGR or greyscale uint8 image as a numpy array.

    Returns:
        Float in [0, 1].  Values below ~0.04 correspond to ``_BLUR_THRESHOLD``
        and typically indicate clinically problematic blur.
    """
    if image is None or image.size == 0:
        raise ValueError("detect_blur requires a non-empty image")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var == 0.0:
        # A uniform / textureless frame has no edges to measure sharpness from.
        # Sharpness is undefined here, not "maximally blurry" — return NaN so
        # callers don't fabricate a 'blur' tag for a contentless image.
        return float("nan")
    score = float(np.clip(lap_var / _BLUR_MAX_VAR, 0.0, 1.0))
    return score


def detect_glare(image: np.ndarray) -> float:
    """Return glare score in [0, 1] where 1 = glare-free, 0 = severe glare.

    Converts the image to HSV and measures the fraction of pixels whose
    Value channel exceeds ``_GLARE_VALUE_THRESHOLD``.  The score is
    ``1 - clamp(fraction / _GLARE_PIXEL_FRACTION, 0, 1)``.

    Args:
        image: BGR uint8 image as a numpy array.  Greyscale images are
            promoted to BGR automatically.

    Returns:
        Float in [0, 1].
    """
    if image is None or image.size == 0:
        return 1.0

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    n_pixels = v_channel.size
    highlight_frac = float(np.count_nonzero(v_channel >= _GLARE_VALUE_THRESHOLD)) / n_pixels
    glare_score = 1.0 - float(np.clip(highlight_frac / _GLARE_PIXEL_FRACTION, 0.0, 1.0))
    return glare_score


def _detect_occlusion(image: np.ndarray) -> float:
    """Return occlusion score in [0, 1] where 1 = unoccluded.

    Estimates occlusion as the fraction of non-dark pixels near the image
    border that might indicate a finger, cheek, or lip encroaching into the
    frame.  The border band is 10% of each dimension.
    """
    if image is None or image.size == 0:
        return 1.0

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape
    bh = max(1, h // 10)
    bw = max(1, w // 10)

    # Extract border pixels
    top = gray[:bh, :]
    bottom = gray[h - bh:, :]
    left = gray[bh: h - bh, :bw]
    right = gray[bh: h - bh, w - bw:]
    border = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])

    # Very dark AND very uniform border → likely fine (dark background)
    # Non-dark (flesh-tone) border → possible occlusion
    flesh_like = float(np.count_nonzero((border > 60) & (border < 200))) / len(border)
    # If >30% of border pixels are flesh-like → probable occlusion
    score = 1.0 - float(np.clip(flesh_like / 0.3, 0.0, 1.0))
    return score


def analyze_view_quality(image_path: str | Path) -> list[str]:
    """Analyse a single view image and return a list of detected quality issues.

    Tags are drawn from ``{"blur", "glare", "occlusion"}``.

    Args:
        image_path: Path to a JPEG or PNG image.

    Returns:
        List of quality-issue tag strings.  An empty list means the image
        passed all checks.  Returns ``["unreadable"]`` if the file cannot
        be opened.
    """
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        logger.warning("Could not read image for quality analysis: %s", path)
        return ["unreadable"]

    tags: list[str] = []

    blur_score = detect_blur(image)
    if np.isnan(blur_score):
        # Sharpness undefined (uniform/textureless frame): low detail, not blur.
        tags.append("low_detail")
    elif blur_score < (_BLUR_THRESHOLD / _BLUR_MAX_VAR):
        tags.append("blur")

    glare_score = detect_glare(image)
    if glare_score < 0.5:
        tags.append("glare")

    occlusion_score = _detect_occlusion(image)
    if occlusion_score < 0.5:
        tags.append("occlusion")

    return tags
