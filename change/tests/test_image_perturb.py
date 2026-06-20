"""Tests for dcc.perturb.image_perturb — numpy and pure-Python perturbation paths."""
from __future__ import annotations

import unittest

import numpy as np

from dcc.perturb.image_perturb import (
    ImagePerturbConfig,
    _apply_numpy,
    apply_image_perturbation,
    random_image_perturb_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_image_np(h: int = 4, w: int = 4) -> np.ndarray:
    """Return a small HxWx3 float image in [0, 1]."""
    rng = np.random.default_rng(0)
    return rng.random((h, w, 3)).astype(float)


# ---------------------------------------------------------------------------
# ImagePerturbConfig tests
# ---------------------------------------------------------------------------

class TestImagePerturbConfig(unittest.TestCase):
    def test_defaults_are_identity(self):
        cfg = ImagePerturbConfig()
        self.assertEqual(cfg.brightness_delta, 0.0)
        self.assertEqual(cfg.contrast_scale, 1.0)
        self.assertEqual(cfg.noise_std, 0.0)
        self.assertFalse(cfg.flip_horizontal)

    def test_config_is_frozen(self):
        cfg = ImagePerturbConfig(brightness_delta=0.1)
        with self.assertRaises((AttributeError, TypeError)):
            cfg.brightness_delta = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# random_image_perturb_config tests
# ---------------------------------------------------------------------------

class TestRandomImagePerturbConfig(unittest.TestCase):
    def test_deterministic_for_same_seed(self):
        c1 = random_image_perturb_config(seed=42)
        c2 = random_image_perturb_config(seed=42)
        self.assertEqual(c1, c2)

    def test_different_seeds_give_different_configs(self):
        c1 = random_image_perturb_config(seed=1)
        c2 = random_image_perturb_config(seed=2)
        self.assertNotEqual(c1, c2)

    def test_parameters_in_valid_ranges(self):
        cfg = random_image_perturb_config(seed=0)
        self.assertGreaterEqual(cfg.brightness_delta, -0.3)
        self.assertLessEqual(cfg.brightness_delta, 0.3)
        self.assertGreaterEqual(cfg.contrast_scale, 0.8)
        self.assertLessEqual(cfg.contrast_scale, 1.2)
        self.assertGreaterEqual(cfg.noise_std, 0.0)
        self.assertLessEqual(cfg.noise_std, 0.05)


# ---------------------------------------------------------------------------
# _apply_numpy tests
# ---------------------------------------------------------------------------

class TestApplyNumpy(unittest.TestCase):
    def test_identity_config_preserves_image(self):
        cfg = ImagePerturbConfig()
        img = _small_image_np()
        result = _apply_numpy(img, cfg)
        np.testing.assert_allclose(result, img, atol=1e-9)

    def test_brightness_shift_adds_delta(self):
        cfg = ImagePerturbConfig(brightness_delta=0.2)
        img = np.zeros((2, 2, 3), dtype=float)
        result = _apply_numpy(img, cfg)
        np.testing.assert_allclose(result, np.full((2, 2, 3), 0.2), atol=1e-9)

    def test_brightness_clamped_to_one(self):
        cfg = ImagePerturbConfig(brightness_delta=0.5)
        img = np.ones((2, 2, 3), dtype=float) * 0.8
        result = _apply_numpy(img, cfg)
        self.assertTrue(np.all(result <= 1.0))

    def test_flip_horizontal_reverses_columns(self):
        cfg = ImagePerturbConfig(flip_horizontal=True)
        img = _small_image_np(w=4)
        result = _apply_numpy(img.copy(), cfg)
        np.testing.assert_allclose(result, img[:, ::-1, :])

    def test_contrast_scale_changes_values(self):
        cfg = ImagePerturbConfig(contrast_scale=1.1)
        img = _small_image_np()
        result = _apply_numpy(img, cfg)
        self.assertFalse(np.allclose(result, img))

    def test_noise_adds_randomness(self):
        cfg = ImagePerturbConfig(noise_std=0.05)
        img = np.full((4, 4, 3), 0.5, dtype=float)
        result = _apply_numpy(img, cfg)
        self.assertFalse(np.allclose(result, img))

    def test_output_clamped_to_zero_one(self):
        cfg = ImagePerturbConfig(brightness_delta=-1.0)
        img = _small_image_np()
        result = _apply_numpy(img, cfg)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))


# ---------------------------------------------------------------------------
# apply_image_perturbation (dispatch) tests
# ---------------------------------------------------------------------------

class TestApplyImagePerturbation(unittest.TestCase):
    def test_numpy_path_returns_ndarray(self):
        cfg = ImagePerturbConfig()
        img = _small_image_np()
        result = apply_image_perturbation(img, cfg)
        self.assertIsInstance(result, np.ndarray)

    def test_output_values_in_range(self):
        cfg = ImagePerturbConfig(brightness_delta=0.3, contrast_scale=1.2, noise_std=0.05)
        img = _small_image_np()
        result = apply_image_perturbation(img, cfg)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))


if __name__ == "__main__":
    unittest.main()
