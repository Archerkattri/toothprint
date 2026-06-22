"""Offscreen dental mesh renderer using Open3D.

Renders dental arch meshes (STL/OBJ/PLY) into the 5 standard protocol
capture views used by DentalMapCert:
  - left_buccal, right_buccal, anterior, upper_occlusal, lower_occlusal

Each view is rendered as a PIL Image (RGB). If open3d is not installed,
raises ImportError with a helpful install message.

2D image-space perturbations (glare, blur) are also provided and operate
on PIL Images.

Usage:
    from toothprint.bench.dmc.render import render_5_views
    views = render_5_views("path/to/arch.stl", resolution=256)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

PROTOCOL_VIEWS = (
    "left_buccal",
    "right_buccal",
    "anterior",
    "upper_occlusal",
    "lower_occlusal",
)


@dataclass(frozen=True)
class RenderedView:
    view_name: str
    width: int
    height: int
    pixels_rgb: bytes  # raw RGB bytes, len = width * height * 3


def _require_open3d():
    try:
        import open3d as o3d

        return o3d
    except ImportError:
        raise ImportError(
            "open3d is required for mesh rendering. Install with: pip install open3d"
        )


def _require_pil():
    try:
        from PIL import Image

        return Image
    except ImportError:
        raise ImportError("Pillow is required. Install with: pip install Pillow")


def _load_mesh(mesh_path: str | Path):
    """Load a mesh from STL/OBJ/PLY using open3d."""
    o3d = _require_open3d()
    path = str(mesh_path)
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) == 0:
        raise ValueError(f"Loaded empty mesh from {path}")
    mesh.compute_vertex_normals()
    return mesh


def _compute_camera_params(mesh, view_name: str, resolution: int) -> dict:
    """Compute camera position and look-at for a named dental protocol view.

    Dental arch coordinate convention (after centering on mesh centroid):
      +Y = up (toward skull)
      +Z = anterior (toward front of face)
      +X = patient's right (our left when facing patient)

    Distance is set to 3× the mesh diagonal for safety.
    """
    import numpy as np

    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center())
    diag = np.linalg.norm(
        np.asarray(bbox.get_max_bound()) - np.asarray(bbox.get_min_bound())
    )
    d = diag * 1.5

    # View directions (camera position relative to centroid)
    offsets = {
        "left_buccal": np.array([-d, 0.0, 0.0]),  # patient's left
        "right_buccal": np.array([d, 0.0, 0.0]),  # patient's right
        "anterior": np.array([0.0, 0.0, d]),  # front
        "upper_occlusal": np.array([0.0, d, 0.0]),  # top-down
        "lower_occlusal": np.array([0.0, -d, 0.0]),  # bottom-up
    }
    up_vectors = {
        "left_buccal": [0.0, 1.0, 0.0],
        "right_buccal": [0.0, 1.0, 0.0],
        "anterior": [0.0, 1.0, 0.0],
        "upper_occlusal": [0.0, 0.0, -1.0],
        "lower_occlusal": [0.0, 0.0, 1.0],
    }

    eye = center + offsets[view_name]
    return {
        "eye": eye.tolist(),
        "center": center.tolist(),
        "up": up_vectors[view_name],
        "width": resolution,
        "height": resolution,
        "fov_deg": 45.0,
    }


def _black_fallback(view_name: str, resolution: int) -> RenderedView:
    """Return a black image as a fallback when the renderer is unavailable."""
    pixels = bytes(resolution * resolution * 3)
    return RenderedView(
        view_name=view_name,
        width=resolution,
        height=resolution,
        pixels_rgb=pixels,
    )


def render_view(
    mesh_path: str | Path, view_name: str, resolution: int = 256
) -> RenderedView:
    """Render a single named protocol view of a dental mesh.

    Parameters
    ----------
    mesh_path : path to STL/OBJ/PLY mesh file
    view_name : one of PROTOCOL_VIEWS
    resolution : output image resolution (square)

    Returns
    -------
    RenderedView with raw RGB bytes
    """
    if view_name not in PROTOCOL_VIEWS:
        raise ValueError(f"Unknown view {view_name!r}. Must be one of {PROTOCOL_VIEWS}")

    o3d = _require_open3d()
    import numpy as np

    mesh = _load_mesh(mesh_path)
    params = _compute_camera_params(mesh, view_name, resolution)

    try:
        # Set up off-screen renderer
        renderer = o3d.visualization.rendering.OffscreenRenderer(resolution, resolution)
        renderer.scene.set_background([0.1, 0.1, 0.1, 1.0])

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry("mesh", mesh, mat)

        renderer.scene.scene.set_sun_light(
            [0.577, 0.577, 0.577], [1.0, 1.0, 1.0], 100000
        )
        renderer.scene.scene.enable_sun_light(True)

        # Set camera
        eye = params["eye"]
        center = params["center"]
        up = params["up"]
        renderer.setup_camera(
            params["fov_deg"],
            center,
            eye,
            up,
        )

        img = renderer.render_to_image()
        img_np = np.asarray(img)

        # Convert to RGB if RGBA
        if img_np.ndim == 3 and img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]

        pixels = bytes(img_np.astype("uint8").tobytes())
        return RenderedView(
            view_name=view_name,
            width=resolution,
            height=resolution,
            pixels_rgb=pixels,
        )
    except Exception as exc:
        warnings.warn(
            f"OffscreenRenderer failed for view '{view_name}' "
            f"(no display / EGL unavailable?): {exc}. "
            "Returning black fallback image.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _black_fallback(view_name, resolution)


def render_5_views(
    mesh_path: str | Path,
    resolution: int = 256,
) -> dict[str, RenderedView]:
    """Render all 5 protocol views of a dental arch mesh.

    Returns dict mapping view_name -> RenderedView.
    """
    return {
        view_name: render_view(mesh_path, view_name, resolution)
        for view_name in PROTOCOL_VIEWS
    }


def rendered_view_to_pil(view: RenderedView):
    """Convert a RenderedView to a PIL Image."""
    Image = _require_pil()
    return Image.frombytes("RGB", (view.width, view.height), view.pixels_rgb)


# ---------------------------------------------------------------------------
# 2D image-space perturbations (operate on PIL Images)
# ---------------------------------------------------------------------------


def apply_glare(image, *, center_frac: float = 0.3, intensity: float = 0.7):
    """Add a circular glare highlight to an image (simulates specular reflection).

    Parameters
    ----------
    image : PIL Image
    center_frac : relative position of glare center as fraction of image size
    intensity : glare brightness in [0, 1]

    Returns
    -------
    PIL Image with glare applied
    """
    import math

    Image = _require_pil()
    img = image.convert("RGB")
    w, h = img.size
    pixels = list(img.getdata())
    cx, cy = int(w * center_frac), int(h * center_frac)
    radius = min(w, h) * 0.15

    new_pixels = []
    for y in range(h):
        for x in range(w):
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist < radius:
                t = intensity * (1.0 - dist / radius) ** 2
                r, g, b = pixels[y * w + x]
                r = min(255, int(r + (255 - r) * t))
                g = min(255, int(g + (255 - g) * t))
                b = min(255, int(b + (255 - b) * t))
                new_pixels.append((r, g, b))
            else:
                new_pixels.append(pixels[y * w + x])

    out = Image.new("RGB", (w, h))
    out.putdata(new_pixels)
    return out


def apply_blur(image, *, sigma: float = 2.0):
    """Apply Gaussian blur to simulate focus error or motion blur.

    Parameters
    ----------
    image : PIL Image
    sigma : Gaussian blur radius in pixels

    Returns
    -------
    Blurred PIL Image
    """
    _require_pil()
    try:
        from PIL import ImageFilter

        return image.filter(ImageFilter.GaussianBlur(radius=sigma))
    except Exception:
        return image
