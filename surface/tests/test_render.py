"""Tests for dentalmapcert.render — 8 tests.

render.py uses lazy imports (open3d is only imported inside functions), so
we patch _require_open3d() per test instead of polluting sys.modules globally.
"""
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dentalmapcert.render import (
    PROTOCOL_VIEWS,
    RenderedView,
    apply_blur,
    apply_glare,
    render_5_views,
    render_view,
    rendered_view_to_pil,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil_image(width: int = 32, height: int = 32, color: tuple = (100, 150, 200)):
    from PIL import Image
    return Image.new("RGB", (width, height), color)


def _make_rendered_view(width: int = 8, height: int = 8, view_name: str = "anterior") -> RenderedView:
    pixels = bytes([128, 64, 32] * width * height)
    return RenderedView(view_name=view_name, width=width, height=height, pixels_rgb=pixels)


def _make_o3d_mock():
    """Return a minimal open3d mock that satisfies render_view."""
    o3d = MagicMock()
    # Mesh mock
    mesh = MagicMock()
    mesh.vertices = [1, 2, 3]
    bbox = MagicMock()
    bbox.get_center.return_value = np.array([0.0, 0.0, 0.0])
    bbox.get_max_bound.return_value = np.array([10.0, 10.0, 10.0])
    bbox.get_min_bound.return_value = np.array([0.0, 0.0, 0.0])
    mesh.get_axis_aligned_bounding_box.return_value = bbox
    o3d.io.read_triangle_mesh.return_value = mesh
    # Renderer mock — render_to_image returns H×W×3 uint8 array
    img_np = np.zeros((8, 8, 3), dtype=np.uint8)
    renderer_mock = MagicMock()
    renderer_mock.render_to_image.return_value = img_np
    o3d.visualization.rendering.OffscreenRenderer.return_value = renderer_mock
    o3d.visualization.rendering.MaterialRecord.return_value = MagicMock()
    return o3d


# ===========================================================================
# Test 1 — PROTOCOL_VIEWS has the 5 expected names
# ===========================================================================

def test_protocol_views_names():
    assert len(PROTOCOL_VIEWS) == 5
    expected = {"left_buccal", "right_buccal", "anterior", "upper_occlusal", "lower_occlusal"}
    assert set(PROTOCOL_VIEWS) == expected


# ===========================================================================
# Test 2 — render_view raises ValueError for unknown view name
# ===========================================================================

def test_render_view_invalid_name_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown view"):
        render_view("dummy.stl", "top_view")


# ===========================================================================
# Test 3 — rendered_view_to_pil returns PIL Image with correct dimensions
# ===========================================================================

def test_rendered_view_to_pil_shape():
    view = _make_rendered_view(width=16, height=16)
    img = rendered_view_to_pil(view)
    assert img.size == (16, 16)
    assert img.mode == "RGB"


# ===========================================================================
# Test 4 — apply_glare does not change image dimensions
# ===========================================================================

def test_apply_glare_returns_same_size():
    img = _make_pil_image(32, 32)
    result = apply_glare(img)
    assert result.size == img.size
    assert result.mode == "RGB"


# ===========================================================================
# Test 5 — apply_blur does not change image dimensions
# ===========================================================================

def test_apply_blur_returns_same_size():
    img = _make_pil_image(32, 32)
    result = apply_blur(img, sigma=1.5)
    assert result.size == img.size


# ===========================================================================
# Test 6 — apply_glare increases brightness at centre pixel
# ===========================================================================

def test_apply_glare_increases_brightness():
    from PIL import Image
    w, h = 64, 64
    img = Image.new("RGB", (w, h), (30, 30, 30))
    result = apply_glare(img, center_frac=0.5, intensity=0.8)
    cx, cy = int(w * 0.5), int(h * 0.5)
    r_orig, g_orig, b_orig = img.getpixel((cx, cy))
    r_new, g_new, b_new = result.getpixel((cx, cy))
    assert r_new > r_orig or g_new > g_orig or b_new > b_orig


# ===========================================================================
# Test 7 — render_5_views returns a dict with all 5 view keys
# ===========================================================================

def test_render_5_views_returns_dict_with_5_keys():
    dummy_view = _make_rendered_view()
    with patch("dentalmapcert.render.render_view", return_value=dummy_view) as mock_rv:
        result = render_5_views("dummy.stl", resolution=8)
    assert set(result.keys()) == set(PROTOCOL_VIEWS)
    assert mock_rv.call_count == 5


# ===========================================================================
# Test 8 — RenderedView.pixels_rgb length == width * height * 3
# ===========================================================================

def test_rendered_view_pixels_length():
    w, h = 12, 7
    pixels = bytes(w * h * 3)
    view = RenderedView(view_name="anterior", width=w, height=h, pixels_rgb=pixels)
    assert len(view.pixels_rgb) == w * h * 3


# ===========================================================================
# Test 9 — _require_open3d returns open3d when available (line 44)
# ===========================================================================

def test_require_open3d_returns_module_when_available():
    """_require_open3d returns the open3d module when it is importable (line 44)."""
    o3d_real = pytest.importorskip("open3d", reason="open3d not installed")
    from dentalmapcert.render import _require_open3d

    result = _require_open3d()
    assert result is o3d_real


# ===========================================================================
# Test 10 — _require_open3d raises ImportError when open3d is blocked (lines 45-49)
# ===========================================================================

def test_require_open3d_raises_when_blocked():
    """_require_open3d raises ImportError when open3d is not importable (lines 45-49)."""
    import sys
    import pytest
    from dentalmapcert.render import _require_open3d

    with patch.dict(sys.modules, {"open3d": None}):
        with pytest.raises(ImportError, match="open3d is required"):
            _require_open3d()


# ===========================================================================
# Test 11 — _require_pil raises ImportError when PIL is blocked (lines 56-57)
# ===========================================================================

def test_require_pil_raises_when_blocked():
    """_require_pil raises ImportError when PIL is not importable (lines 56-57)."""
    import sys
    import pytest
    from dentalmapcert.render import _require_pil

    with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
        with pytest.raises(ImportError, match="Pillow is required"):
            _require_pil()


# ===========================================================================
# Test 12 — _black_fallback returns RenderedView with correct size (lines 114-121)
# ===========================================================================

def test_black_fallback_returns_correct_size():
    """_black_fallback returns a RenderedView with all-zero pixels (lines 114-121)."""
    from dentalmapcert.render import _black_fallback

    view = _black_fallback("anterior", 8)
    assert view.view_name == "anterior"
    assert view.width == 8
    assert view.height == 8
    assert len(view.pixels_rgb) == 8 * 8 * 3
    assert all(b == 0 for b in view.pixels_rgb)


# ===========================================================================
# Test 13 — apply_blur exception path (lines 278-279)
# ===========================================================================

def test_apply_blur_exception_path_returns_image():
    """apply_blur returns the original image when filter raises (lines 278-279)."""
    img = MagicMock()
    img.filter.side_effect = RuntimeError("filter failed")
    with patch("dentalmapcert.render._require_pil"):
        result = apply_blur(img)
    assert result is img


# ===========================================================================
# Test 14 — render_view success using mock open3d (covers lines 62-68, 81-105, 142-177)
# ===========================================================================

def test_render_view_with_mock_o3d_returns_rendered_view():
    """render_view returns a RenderedView when open3d mock succeeds (lines 62-186)."""
    o3d_mock = _make_o3d_mock()
    with patch("dentalmapcert.render._require_open3d", return_value=o3d_mock):
        result = render_view("dummy.stl", "anterior", resolution=8)
    assert isinstance(result, RenderedView)
    assert result.view_name == "anterior"
    assert result.width == 8
    assert result.height == 8


# ===========================================================================
# Test 15 — _load_mesh raises ValueError for empty mesh (line 66)
# ===========================================================================

def test_load_mesh_raises_on_empty_vertices():
    """_load_mesh raises ValueError when mesh has 0 vertices (line 66)."""
    import pytest
    from dentalmapcert.render import _load_mesh

    o3d_mock = MagicMock()
    empty_mesh = MagicMock()
    empty_mesh.vertices = []  # empty → triggers line 66
    o3d_mock.io.read_triangle_mesh.return_value = empty_mesh

    with patch("dentalmapcert.render._require_open3d", return_value=o3d_mock):
        with pytest.raises(ValueError, match="empty mesh"):
            _load_mesh("dummy.stl")


# ===========================================================================
# Test 16 — render_view RGBA conversion (line 178)
# ===========================================================================

def test_render_view_rgba_image_is_converted_to_rgb():
    """render_view strips the alpha channel when render_to_image returns RGBA (line 178)."""
    o3d_mock = _make_o3d_mock()
    # Override the render_to_image return to 4-channel RGBA
    rgba_img = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba_img[:, :, :3] = 100  # RGB channels non-zero
    renderer_mock = o3d_mock.visualization.rendering.OffscreenRenderer.return_value
    renderer_mock.render_to_image.return_value = rgba_img

    with patch("dentalmapcert.render._require_open3d", return_value=o3d_mock):
        result = render_view("dummy.stl", "anterior", resolution=8)
    assert isinstance(result, RenderedView)
    assert len(result.pixels_rgb) == 8 * 8 * 3  # 3 channels, not 4


# ===========================================================================
# Test 17 — render_view exception path returns black fallback (lines 187-195)
# ===========================================================================

def test_render_view_exception_returns_black_fallback():
    """render_view returns a black fallback when OffscreenRenderer raises (lines 187-195)."""
    import warnings
    o3d_mock = _make_o3d_mock()
    renderer_mock = o3d_mock.visualization.rendering.OffscreenRenderer.return_value
    renderer_mock.render_to_image.side_effect = RuntimeError("no display")

    with patch("dentalmapcert.render._require_open3d", return_value=o3d_mock):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = render_view("dummy.stl", "anterior", resolution=8)
    assert isinstance(result, RenderedView)
    assert result.view_name == "anterior"
    assert all(b == 0 for b in result.pixels_rgb)
