"""Tests for the reconstruction module (FIX 9)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _resolve_device
# ---------------------------------------------------------------------------


def test_resolve_device_auto_returns_string():
    from dentalmapcert.reconstruction import _resolve_device

    result = _resolve_device("auto")
    assert isinstance(result, str)
    assert result in ("cuda", "cpu")


def test_resolve_device_explicit_cpu():
    from dentalmapcert.reconstruction import _resolve_device

    assert _resolve_device("cpu") == "cpu"


def test_resolve_device_explicit_cuda():
    from dentalmapcert.reconstruction import _resolve_device

    # Just verifies the value passes through unchanged (no auto-resolution).
    assert _resolve_device("cuda") == "cuda"


def test_resolve_device_auto_with_torch_blocked_returns_cpu():
    """_resolve_device returns 'cpu' when torch is not importable (lines 34-36)."""
    import sys
    from unittest.mock import patch
    from dentalmapcert.reconstruction import _resolve_device

    with patch.dict(sys.modules, {"torch": None}):
        result = _resolve_device("auto")
    assert result == "cpu"


# ---------------------------------------------------------------------------
# reconstruct_point_cloud — run-or-raise (no fallback, no fall-through)
# ---------------------------------------------------------------------------


def test_reconstruct_point_cloud_returns_vggt_result():
    """backend='vggt' returns the VGGT reconstruction when it succeeds."""
    from unittest.mock import patch
    from dentalmapcert.reconstruction import reconstruct_point_cloud

    fake_pts = np.ones((10, 3), dtype=np.float64)
    fake_conf = np.ones(10, dtype=np.float64)
    with patch("dentalmapcert.reconstruction._reconstruct_vggt", return_value=(fake_pts, fake_conf)):
        pts, conf = reconstruct_point_cloud([Path("a.png")], backend="vggt")
    np.testing.assert_array_equal(pts, fake_pts)


def test_reconstruct_point_cloud_returns_dust3r_result():
    """backend='dust3r' runs DUSt3R directly (no implicit cascade from vggt)."""
    from unittest.mock import patch
    from dentalmapcert.reconstruction import reconstruct_point_cloud

    fake_pts = np.ones((5, 3), dtype=np.float64)
    fake_conf = np.ones(5, dtype=np.float64)
    with patch("dentalmapcert.reconstruction._reconstruct_dust3r", return_value=(fake_pts, fake_conf)):
        pts, conf = reconstruct_point_cloud([Path("a.png"), Path("b.png")], backend="dust3r")
    np.testing.assert_array_equal(pts, fake_pts)


def test_reconstruct_point_cloud_raises_when_backend_fails():
    """A backend that cannot reconstruct RAISES — there is no crude fallback."""
    from unittest.mock import patch
    from dentalmapcert.reconstruction import reconstruct_point_cloud

    with patch("dentalmapcert.reconstruction._reconstruct_vggt", return_value=None):
        with pytest.raises(RuntimeError, match="failed to produce a reconstruction"):
            reconstruct_point_cloud([Path("a.png")], backend="vggt")


def test_reconstruct_point_cloud_does_not_cascade_vggt_to_dust3r():
    """When VGGT fails, DUSt3R is NOT silently tried — the call raises instead."""
    from unittest.mock import patch
    from dentalmapcert.reconstruction import reconstruct_point_cloud

    with patch("dentalmapcert.reconstruction._reconstruct_vggt", return_value=None), \
         patch("dentalmapcert.reconstruction._reconstruct_dust3r") as dust3r:
        with pytest.raises(RuntimeError):
            reconstruct_point_cloud([Path("a.png")], backend="vggt")
        dust3r.assert_not_called()


def test_reconstruct_point_cloud_empty_input_raises():
    from dentalmapcert.reconstruction import reconstruct_point_cloud

    with pytest.raises(ValueError, match="at least one image"):
        reconstruct_point_cloud([])


def test_reconstruct_point_cloud_rejects_unknown_backend():
    from dentalmapcert.reconstruction import reconstruct_point_cloud

    with pytest.raises(ValueError, match="backend must be one of"):
        reconstruct_point_cloud([Path("a.png")], backend="not_a_backend")


def test_open3d_backend_no_longer_exists():
    """The crude open3d edge-projection fallback was removed entirely."""
    from dentalmapcert import reconstruction

    assert "open3d" not in reconstruction.BACKENDS
    assert not hasattr(reconstruction, "_reconstruct_open3d_fallback")


# ---------------------------------------------------------------------------
# VGGT backend — returns None without the package
# ---------------------------------------------------------------------------


def test_vggt_backend_returns_none_without_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """If vggt is not installed, _reconstruct_vggt must return None gracefully."""
    # Force ImportError for the vggt package.
    monkeypatch.setitem(sys.modules, "vggt", None)
    monkeypatch.setitem(sys.modules, "vggt.models", None)
    monkeypatch.setitem(sys.modules, "vggt.models.vggt", None)

    # Re-import after patching so the import guard at function-call time fires.
    import dentalmapcert.reconstruction as recon_mod
    importlib.reload(recon_mod)

    result = recon_mod._reconstruct_vggt([], "cpu")
    assert result is None


# ---------------------------------------------------------------------------
# DUSt3R backend — returns None without the package
# ---------------------------------------------------------------------------


def test_dust3r_backend_returns_none_without_package(monkeypatch: pytest.MonkeyPatch):
    """If mini_dust3r is not installed, _reconstruct_dust3r must return None gracefully."""
    # Patch the module-level _dust3r_infer sentinel to None.
    import dentalmapcert.reconstruction as recon_mod

    monkeypatch.setattr(recon_mod, "_dust3r_infer", None)

    result = recon_mod._reconstruct_dust3r([], "cpu")
    assert result is None


# ---------------------------------------------------------------------------
# VGGT backend — mocked inference paths (lines 59-138)
# ---------------------------------------------------------------------------


def test_reconstruct_vggt_model_load_failure_returns_none(tmp_path: Path):
    """Lines 59-66: VGGT model load failure → return None."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    from unittest.mock import MagicMock, patch
    from PIL import Image
    import dentalmapcert.reconstruction as recon_mod

    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    img_path = tmp_path / "img.png"
    img.save(str(img_path))

    fake_class = MagicMock()
    fake_class.from_pretrained.side_effect = RuntimeError("model unavailable")

    with patch("vggt.models.vggt.VGGT", fake_class):
        result = recon_mod._reconstruct_vggt([img_path], "cpu")
    assert result is None


def test_reconstruct_vggt_no_world_points_returns_none(tmp_path: Path):
    """Lines 100-102: model returns no world_points key → return None."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    from unittest.mock import MagicMock, patch
    from PIL import Image
    import dentalmapcert.reconstruction as recon_mod

    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    img_path = tmp_path / "img.png"
    img.save(str(img_path))

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    fake_model.return_value = {}  # no world_points or pts3d

    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = fake_model

    with patch("vggt.models.vggt.VGGT", fake_class):
        result = recon_mod._reconstruct_vggt([img_path], "cpu")
    assert result is None


def test_reconstruct_vggt_success_with_mocked_model(tmp_path: Path):
    """Lines 59-138: full success path through VGGT inference with mocked model.

    Uses 'pts3d'/'conf' keys (not 'world_points'/'world_points_conf') so that
    the `or` expression evaluates None on the left side and returns the tensor
    without calling bool() on a multi-element tensor.
    """
    torch = pytest.importorskip("torch", reason="torch not installed")
    from unittest.mock import MagicMock, patch
    from PIL import Image
    import dentalmapcert.reconstruction as recon_mod

    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    img_path = tmp_path / "img.png"
    img.save(str(img_path))

    # Use pts3d + conf keys: predictions.get("world_points") → None,
    # then None or predictions.get("pts3d") → tensor (no bool(tensor) call).
    # dim=5 tensor so the wp.dim()==5 branch runs.
    fake_world_pts = torch.ones(1, 1, 8, 8, 3, dtype=torch.float32)
    fake_conf_t = torch.ones(1, 1, 8, 8, dtype=torch.float32)

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    fake_model.return_value = {"pts3d": fake_world_pts, "conf": fake_conf_t}

    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = fake_model

    with patch("vggt.models.vggt.VGGT", fake_class):
        result = recon_mod._reconstruct_vggt([img_path], "cpu")

    assert result is not None
    pts, conf = result
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert conf.ndim == 1


def test_reconstruct_vggt_no_conf_uses_uniform(tmp_path: Path):
    """Lines 116-117: when no conf key in output, fallback to uniform ones."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    from unittest.mock import MagicMock, patch
    from PIL import Image
    import dentalmapcert.reconstruction as recon_mod

    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    img_path = tmp_path / "img.png"
    img.save(str(img_path))

    # pts3d without conf → fallback uniform confidence.
    # dim=4 (no batch dim) so the wp.dim()==5 branch is NOT taken.
    fake_world_pts = torch.ones(1, 8, 8, 3, dtype=torch.float32)

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    fake_model.return_value = {"pts3d": fake_world_pts}  # no conf at all

    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = fake_model

    with patch("vggt.models.vggt.VGGT", fake_class):
        result = recon_mod._reconstruct_vggt([img_path], "cpu")

    assert result is not None


def test_reconstruct_vggt_inference_exception_returns_none(tmp_path: Path):
    """Lines 136-138: inference exception → return None (outer try/except)."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    from unittest.mock import MagicMock, patch
    from PIL import Image
    import dentalmapcert.reconstruction as recon_mod

    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    img_path = tmp_path / "img.png"
    img.save(str(img_path))

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    fake_model.side_effect = RuntimeError("inference crashed")

    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = fake_model

    with patch("vggt.models.vggt.VGGT", fake_class):
        result = recon_mod._reconstruct_vggt([img_path], "cpu")
    assert result is None


def test_reconstruct_vggt_cuda_uses_autocast(tmp_path: Path):
    """On a 'cuda' device the VGGT forward runs under torch.autocast (mixed
    precision) instead of manually halving the model + input — which raised a
    dtype mismatch because VGGT re-casts tensors to float32 internally."""
    torch = pytest.importorskip("torch", reason="torch not installed")
    from contextlib import contextmanager
    from unittest.mock import MagicMock, patch
    from PIL import Image
    import dentalmapcert.reconstruction as recon_mod

    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    img_path = tmp_path / "img.png"
    img.save(str(img_path))

    fake_world_pts = torch.ones(1, 8, 8, 3, dtype=torch.float32)
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    fake_model.return_value = {"pts3d": fake_world_pts}
    # model.half() must NOT be called any more.
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = fake_model

    fake_images_t = MagicMock()
    fake_images_t.unsqueeze.return_value = fake_images_t
    fake_images_t.to.return_value = fake_images_t

    autocast_used = {"entered": False, "kwargs": None}

    @contextmanager
    def _fake_autocast(*args, **kwargs):
        autocast_used["entered"] = True
        autocast_used["kwargs"] = kwargs
        yield

    with patch("vggt.models.vggt.VGGT", fake_class), \
         patch("torch.from_numpy", return_value=fake_images_t), \
         patch("torch.autocast", _fake_autocast):
        recon_mod._reconstruct_vggt([img_path], "cuda")

    assert autocast_used["entered"], "cuda VGGT forward must run under torch.autocast"
    assert autocast_used["kwargs"].get("device_type") == "cuda"
    fake_model.half.assert_not_called()


# ---------------------------------------------------------------------------
# DUSt3R backend — mocked inference paths (lines 164-202)
# ---------------------------------------------------------------------------


def test_reconstruct_dust3r_croco_import_fails_returns_none():
    """Lines 164-166: AsymmetricCroCo3DStereo import error → return None."""
    import sys
    from unittest.mock import patch
    import dentalmapcert.reconstruction as recon_mod

    saved_infer = recon_mod._dust3r_infer
    recon_mod._dust3r_infer = object()  # non-None so we don't skip early

    try:
        with patch.dict(sys.modules, {"mini_dust3r.model": None}):
            result = recon_mod._reconstruct_dust3r(["img1.png", "img2.png"], "cpu")
    finally:
        recon_mod._dust3r_infer = saved_infer

    assert result is None


def test_reconstruct_dust3r_fewer_than_2_images_returns_none():
    """Lines 168-170: fewer than 2 images → return None."""
    pytest.importorskip("mini_dust3r", reason="mini_dust3r not installed")
    import dentalmapcert.reconstruction as recon_mod
    from unittest.mock import MagicMock, patch

    saved_infer = recon_mod._dust3r_infer
    recon_mod._dust3r_infer = MagicMock()

    fake_croco = MagicMock()
    try:
        with patch("mini_dust3r.model.AsymmetricCroCo3DStereo", fake_croco):
            result = recon_mod._reconstruct_dust3r(["only_one.png"], "cpu")
    finally:
        recon_mod._dust3r_infer = saved_infer

    assert result is None


def test_reconstruct_dust3r_model_load_failure_returns_none():
    """Lines 172-180: DUSt3R model load exception → return None."""
    pytest.importorskip("mini_dust3r", reason="mini_dust3r not installed")
    import dentalmapcert.reconstruction as recon_mod
    from unittest.mock import MagicMock, patch

    saved_infer = recon_mod._dust3r_infer
    recon_mod._dust3r_infer = MagicMock()

    fake_croco = MagicMock()
    fake_croco.from_pretrained.side_effect = RuntimeError("no model")
    try:
        with patch("mini_dust3r.model.AsymmetricCroCo3DStereo", fake_croco):
            result = recon_mod._reconstruct_dust3r(["a.png", "b.png"], "cpu")
    finally:
        recon_mod._dust3r_infer = saved_infer

    assert result is None


def test_reconstruct_dust3r_success_with_mocked_model():
    """Lines 182-199: full success path through DUSt3R inference with mocked model."""
    pytest.importorskip("mini_dust3r", reason="mini_dust3r not installed")
    import dentalmapcert.reconstruction as recon_mod
    from unittest.mock import MagicMock, patch

    fake_pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    fake_result = MagicMock()
    fake_result.point_cloud.vertices = fake_pts
    fake_result.conf = None  # triggers uniform-conf fallback

    fake_infer = MagicMock(return_value=fake_result)
    saved_infer = recon_mod._dust3r_infer
    recon_mod._dust3r_infer = fake_infer

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model

    fake_croco = MagicMock()
    fake_croco.from_pretrained.return_value = fake_model

    try:
        with patch("mini_dust3r.model.AsymmetricCroCo3DStereo", fake_croco):
            result = recon_mod._reconstruct_dust3r(["a.png", "b.png"], "cpu")
    finally:
        recon_mod._dust3r_infer = saved_infer

    assert result is not None
    pts, conf = result
    assert pts.shape == (2, 3)
    assert len(conf) == 2


def test_reconstruct_dust3r_success_with_real_conf():
    """Lines 194-195: DUSt3R result includes conf array → clip and return."""
    pytest.importorskip("mini_dust3r", reason="mini_dust3r not installed")
    import dentalmapcert.reconstruction as recon_mod
    from unittest.mock import MagicMock, patch

    fake_pts = np.ones((3, 3))
    fake_conf_arr = np.array([0.9, 1.5, 0.2])  # values above 1.0 clipped

    fake_result = MagicMock()
    fake_result.point_cloud.vertices = fake_pts
    fake_result.conf = fake_conf_arr

    fake_infer = MagicMock(return_value=fake_result)
    saved_infer = recon_mod._dust3r_infer
    recon_mod._dust3r_infer = fake_infer

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model

    fake_croco = MagicMock()
    fake_croco.from_pretrained.return_value = fake_model

    try:
        with patch("mini_dust3r.model.AsymmetricCroCo3DStereo", fake_croco):
            result = recon_mod._reconstruct_dust3r(["a.png", "b.png"], "cpu")
    finally:
        recon_mod._dust3r_infer = saved_infer

    assert result is not None
    pts, conf = result
    assert float(conf.max()) <= 1.0  # clipped


def test_reconstruct_dust3r_inference_exception_returns_none():
    """Lines 200-202: DUSt3R inference exception → return None."""
    pytest.importorskip("mini_dust3r", reason="mini_dust3r not installed")
    import dentalmapcert.reconstruction as recon_mod
    from unittest.mock import MagicMock, patch

    fake_infer = MagicMock(side_effect=RuntimeError("inference failed"))
    saved_infer = recon_mod._dust3r_infer
    recon_mod._dust3r_infer = fake_infer

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model

    fake_croco = MagicMock()
    fake_croco.from_pretrained.return_value = fake_model

    try:
        with patch("mini_dust3r.model.AsymmetricCroCo3DStereo", fake_croco):
            result = recon_mod._reconstruct_dust3r(["a.png", "b.png"], "cpu")
    finally:
        recon_mod._dust3r_infer = saved_infer

    assert result is None


# ---------------------------------------------------------------------------
# Module-level mini_dust3r fallback import (lines 147-151)
# ---------------------------------------------------------------------------


def test_dust3r_module_fallback_import_when_typo_missing(monkeypatch: pytest.MonkeyPatch):
    """Lines 147-149: when inferece_dust3r (typo) absent, use inference_dust3r spelling."""
    import importlib
    import sys
    import types
    from unittest.mock import MagicMock
    import dentalmapcert.reconstruction as recon_mod

    # Fake module has ONLY the correct spelling (no typo).
    # Reload triggers: line 146 ImportError → line 147 except → line 149 succeeds.
    fake_infer_fn = MagicMock()
    fake_api_mod = types.ModuleType("mini_dust3r.api.inference")
    fake_api_mod.inference_dust3r = fake_infer_fn

    monkeypatch.setitem(sys.modules, "mini_dust3r.api.inference", fake_api_mod)
    importlib.reload(recon_mod)
    assert recon_mod._dust3r_infer is fake_infer_fn


def test_dust3r_module_both_spellings_missing_gives_none(monkeypatch: pytest.MonkeyPatch):
    """Lines 150-151: both import spellings fail → _dust3r_infer = None."""
    import importlib
    import sys
    import types
    import dentalmapcert.reconstruction as recon_mod

    # Fake module has NEITHER spelling → both imports raise ImportError.
    fake_api_mod = types.ModuleType("mini_dust3r.api.inference")
    # No functions set at all.

    monkeypatch.setitem(sys.modules, "mini_dust3r.api.inference", fake_api_mod)
    importlib.reload(recon_mod)
    assert recon_mod._dust3r_infer is None
