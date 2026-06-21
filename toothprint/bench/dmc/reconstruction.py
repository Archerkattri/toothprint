"""3D reconstruction adapter for DentalMapCert.

Tries VGGT (Meta FAIR, CVPR 2025) first, then falls back to MASt3R/DUSt3R
via mini-dust3r.  If neither GPU model is available or images are insufficient,
a lightweight Open3D ICP fallback is used.

The public API is a single function:

    points_xyz, confidence = reconstruct_point_cloud(image_paths)

Both outputs are numpy arrays; ``confidence`` is per-point and in [0, 1].
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# VGGT backend (Meta FAIR, CVPR 2025)
# ---------------------------------------------------------------------------

def _reconstruct_vggt(
    image_paths: list[Path],
    device: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Attempt VGGT reconstruction.  Returns None if unavailable or failed."""
    try:
        import torch
        from PIL import Image as PILImage
        from vggt.models.vggt import VGGT
    except ImportError:
        logger.debug("VGGT not importable; skipping VGGT backend.")
        return None

    try:
        logger.info("Loading VGGT-1B from HuggingFace (facebook/VGGT-1B)…")
        model = VGGT.from_pretrained("facebook/VGGT-1B")
        model = model.to(device)
        model.eval()
    except Exception as exc:
        logger.warning("VGGT model load failed (%s); skipping VGGT backend.", exc)
        return None

    try:
        # Preprocess: resize to 518×518 (VGGT default), normalise to [0,1]
        target_size = 518
        frames: list[np.ndarray] = []
        for p in image_paths:
            img = PILImage.open(p).convert("RGB").resize(
                (target_size, target_size), PILImage.BILINEAR
            )
            frames.append(np.array(img, dtype=np.float32) / 255.0)

        # Shape: (S, 3, H, W)
        images_np = np.stack(frames, axis=0).transpose(0, 3, 1, 2)

        # Add batch dimension: (1, S, 3, H, W)
        images_t = torch.from_numpy(images_np).unsqueeze(0).to(device)

        # Use autocast (mixed precision) on CUDA for memory efficiency rather
        # than manually halving the model + input: VGGT re-casts tensors to
        # float32 internally, so a half-weight model on a manually-halved input
        # raises a dtype mismatch. autocast keeps weights in float32 and casts
        # the ops it can to float16, which is correct and avoids the mismatch.
        with torch.no_grad():
            # VGGT forward expects (B, S, 3, H, W)
            if "cuda" in device:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    predictions = model(images_t)
            else:
                predictions = model(images_t)

        # VGGT outputs: world_points (B, S, H, W, 3) or (S, H, W, 3)
        # Try both key names for resilience across VGGT versions
        world_points_raw = predictions.get("world_points") or predictions.get("pts3d")
        conf_raw = predictions.get("world_points_conf") or predictions.get("conf")

        if world_points_raw is None:
            logger.warning("VGGT: no world_points/pts3d key in output; falling through.")
            return None

        # Handle batch dim: if (B, S, H, W, 3) take [0]; if (S, H, W, 3) use as-is
        wp = world_points_raw
        if wp.dim() == 5:
            wp = wp[0]  # (S, H, W, 3)

        world_points = wp.cpu().numpy()

        if conf_raw is not None:
            cf = conf_raw
            if cf.dim() == 4:
                cf = cf[0]  # (S, H, W)
            conf = cf.cpu().numpy()
        else:
            conf = np.ones(world_points.shape[:3], dtype=np.float32)

        # Flatten all views into a single point cloud
        pts = world_points.reshape(-1, 3)
        conf_flat = conf.reshape(-1)

        # Keep only confident points (above median confidence)
        thresh = float(np.median(conf_flat))
        mask = conf_flat >= thresh
        pts_filtered = pts[mask]
        conf_filtered = conf_flat[mask]

        # Normalise confidence to [0, 1]
        c_max = float(conf_filtered.max()) if conf_filtered.size > 0 else 1.0
        conf_norm = np.clip(conf_filtered / max(c_max, 1e-8), 0.0, 1.0)

        logger.info("VGGT: produced %d points from %d views.", len(pts_filtered), len(image_paths))
        return pts_filtered.astype(np.float64), conf_norm.astype(np.float64)

    except Exception as exc:
        logger.warning("VGGT inference failed (%s); falling through.", exc)
        return None


# ---------------------------------------------------------------------------
# MASt3R/DUSt3R backend via mini-dust3r
# ---------------------------------------------------------------------------

try:
    from mini_dust3r.api.inference import inferece_dust3r as _dust3r_infer  # type: ignore[import]
except ImportError:
    try:
        from mini_dust3r.api.inference import inference_dust3r as _dust3r_infer  # type: ignore[import]
    except ImportError:
        _dust3r_infer = None  # type: ignore[assignment]


def _reconstruct_dust3r(
    image_paths: list[Path],
    device: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Attempt DUSt3R reconstruction.  Returns None if unavailable or failed."""
    if _dust3r_infer is None:
        logger.debug("mini_dust3r not importable; skipping DUSt3R backend.")
        return None
    try:
        from mini_dust3r.model import AsymmetricCroCo3DStereo  # type: ignore[import]
    except ImportError:
        logger.debug("mini_dust3r not importable; skipping DUSt3R backend.")
        return None

    if len(image_paths) < 2:
        logger.debug("DUSt3R requires ≥2 images; skipping.")
        return None

    try:
        logger.info("Loading DUSt3R model from HuggingFace…")
        model = AsymmetricCroCo3DStereo.from_pretrained(
            "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        ).to(device)
        model.eval()
    except Exception as exc:
        logger.warning("DUSt3R model load failed (%s); skipping.", exc)
        return None

    try:
        result = _dust3r_infer(
            image_dir_or_list=[str(p) for p in image_paths],
            model=model,
            device=device,
            batch_size=1,
            image_size=512,
            niter=100,
        )
        pts = np.asarray(result.point_cloud.vertices, dtype=np.float64)
        # Use real per-point confidence if available from mini-dust3r output;
        # otherwise fall back to a uniform 0.8 placeholder.
        if hasattr(result, 'conf') and result.conf is not None:
            conf = np.clip(np.asarray(result.conf, dtype=np.float64), 0.0, 1.0)
        else:
            conf = np.full(len(pts), 0.8, dtype=np.float64)
        logger.info("DUSt3R: produced %d points.", len(pts))
        return pts, conf
    except Exception as exc:
        logger.warning("DUSt3R inference failed (%s); falling through.", exc)
        return None



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

BACKENDS = ("vggt", "dust3r")


def reconstruct_point_cloud(
    image_paths: list[Path],
    device: str = "auto",
    backend: str = "vggt",
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a point cloud from unposed smartphone images.

    The ``backend`` selects the metric neural reconstructor and is run
    strictly: there is NO fallback and NO silent cascade between backends.

    - ``"vggt"``   — VGGT (default). Raises if VGGT cannot run.
    - ``"dust3r"`` — DUSt3R via mini-dust3r. Raises if DUSt3R cannot run.

    Args:
        image_paths: Ordered list of image paths (any pose; unposed is fine).
        device:      ``"cuda"``, ``"cpu"``, or ``"auto"`` (default).
        backend:     One of :data:`BACKENDS` (default ``"vggt"``).

    Returns:
        Tuple of:
            - ``points_xyz``: ``np.ndarray`` of shape ``(N, 3)`` at the backend's
              native/up-to-scale units; callers using these for metric error
              must rescale to mm first.
            - ``confidence``: ``np.ndarray`` of shape ``(N,)`` in ``[0, 1]``.

    Raises:
        ValueError:   if ``backend`` is unknown or ``image_paths`` is empty.
        RuntimeError: if the selected backend fails to produce a reconstruction.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if not image_paths:
        raise ValueError("reconstruct_point_cloud requires at least one image path")

    dev = _resolve_device(device)
    logger.info(
        "reconstruct_point_cloud: %d images on device=%s backend=%s",
        len(image_paths), dev, backend,
    )

    # Run the requested backend or FAIL — no silent fall-through, no crude CPU
    # fallback. If a backend can't run, the caller must fix the real cause
    # (install it, free GPU VRAM, supply >=2 usable views), not get a degraded
    # uncalibrated result that looks like a metric reconstruction.
    runner = {"vggt": _reconstruct_vggt, "dust3r": _reconstruct_dust3r}[backend]
    result = runner(image_paths, dev)
    if result is None:
        raise RuntimeError(
            f"backend={backend!r} failed to produce a reconstruction. No fallback "
            f"is provided by design: fix the real cause (e.g. install the backend, "
            f"use a GPU with enough VRAM, or pass >=2 usable views)."
        )
    return result
