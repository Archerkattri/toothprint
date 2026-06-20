import numpy as np
import pytest

from toothprint.geometry import distance, mean_point, umeyama


def test_mean_point():
    assert np.allclose(mean_point([[0, 0], [2, 4]]), [1, 2])


def test_mean_point_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean_point([])


def test_distance():
    assert distance([0, 0], [3, 4]) == 5.0


def test_umeyama_recovers_similarity_3d():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(30, 3))
    ang = 0.5
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    dst = 2.3 * src @ R.T + np.array([1.0, -2.0, 3.0])
    A, t = umeyama(src, dst)
    assert np.allclose(src @ A.T + t, dst, atol=1e-6)


def test_umeyama_rigid_only():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(20, 2))
    A, t = umeyama(src, 5.0 * src, with_scale=False)
    # rigid fit cannot absorb the 5x scale -> mapped points are not 5x.
    assert not np.allclose(src @ A.T + t, 5.0 * src, atol=1e-3)
