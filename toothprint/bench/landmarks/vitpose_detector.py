"""ViTPose dental landmark detector (Xu et al., NeurIPS 2022).

Uses the real pose-pretrained ViTPose from HuggingFace
(``usyd-community/vitpose-base-simple``) — a plain ViT backbone with a simple
deconvolution heatmap decoder — and retargets its head from 17 COCO human-pose
keypoints to the 5 dental landmarks. Heatmap regression is the state-of-the-art
for precise keypoint localisation and replaces the dated KeypointRCNN
(Faster-RCNN, 2017) box-then-regress keypoint head.

Top-down: each tooth is cropped to ViTPose's 256x192 input, the network predicts
K heatmaps, and the sub-pixel argmax is mapped back to image pixels.

Landmarks (K=5): cej_left, cej_right, crest_mesial, crest_distal, apex.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LANDMARK_NAMES = ("cej_left", "cej_right", "crest_mesial", "crest_distal", "apex")
NUM_LANDMARKS = len(LANDMARK_NAMES)
INPUT_H, INPUT_W = 256, 192  # ViTPose canonical input (H, W)
PRETRAINED_ID = "usyd-community/vitpose-base-simple"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def tooth_to_landmarks(tooth: dict):
    """Map one tooth annotation to (pts[5, 2], vis[5]) in image pixels.

    The 5 ordered landmarks are cej_left, cej_right, crest_mesial, crest_distal,
    apex (see ``LANDMARK_NAMES``). Absent landmarks are NaN with vis=False.
    """
    cej = tooth.get("cej") or []
    crest = tooth.get("crest_line") or []
    apex = tooth.get("apex") or []
    pts = np.full((NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
    if len(cej) >= 1:
        pts[0] = cej[0]
    if len(cej) >= 2:
        pts[1] = cej[1]
    if len(crest) >= 1:
        pts[2] = crest[0]
    if len(crest) >= 2:
        pts[3] = crest[-1]
    if len(apex) >= 1:
        pts[4] = apex[0]
    vis = ~np.isnan(pts).any(axis=1)
    return pts, vis


def landmark_box(pts: np.ndarray, vis: np.ndarray, w: int, h: int):
    """Padded square-ish crop box [x1, y1, x2, y2] around the visible landmarks.

    Mirrors the top-down training crop so inference sees the same framing.
    """
    v = pts[vis]
    x1, y1 = v.min(axis=0)
    x2, y2 = v.max(axis=0)
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    margin = 0.45 * max(bw, bh) + 15.0
    x1, y1, x2, y2 = x1 - margin, y1 - margin, x2 + margin, y2 + margin
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(w), x2), min(float(h), y2)
    return float(x1), float(y1), float(x2), float(y2)


def predict_tooth(
    detector, image_rgb, tooth, *, n_tta: int = 1, jitter_px: int = 4, rng=None
):
    """Run ``detector`` on one tooth crop of ``image_rgb`` (HxWx3 uint8).

    Crops to the GT landmark box, runs top-down inference, and maps the crop-space
    landmarks back to image pixels. Returns a predicted tooth dict mirroring the
    GT tooth's visible-landmark structure (same ``tooth_id``, predicted coords),
    or ``None`` if the tooth has no visible landmarks or the crop is degenerate.

    With ``n_tta > 1`` the crop box is randomly jittered ``n_tta`` times and the
    per-landmark **median** of the back-mapped predictions is returned. This
    test-time augmentation suppresses the heatmap argmax-flip outliers that make
    single-shot ViTPose unreliable (bimodal heatmaps where a 1-px perturbation
    flips the peak), at K× the inference cost.

    ``detector`` is any object exposing ``predict_crop(crop_rgb) -> [[x, y], ...]``.
    """
    pts, vis = tooth_to_landmarks(tooth)
    if not vis.any():
        return None
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = landmark_box(pts, vis, w, h)
    ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
    if (ix2 - ix1) < 2 or (iy2 - iy1) < 2:
        return None

    if rng is None:
        rng = np.random.default_rng(0)
    offsets = [(0, 0)]
    for _ in range(max(0, n_tta - 1)):
        offsets.append(
            (
                int(rng.integers(-jitter_px, jitter_px + 1)),
                int(rng.integers(-jitter_px, jitter_px + 1)),
            )
        )

    samples = []  # (n_aug, K, 2) in image coords
    for ox, oy in offsets:
        # Clamp keeps each jittered box >= 2px in both dims, so the crop is valid.
        ax1 = min(max(0, ix1 + ox), w - 2)
        ay1 = min(max(0, iy1 + oy), h - 2)
        ax2 = min(max(ax1 + 2, ix2 + ox), w)
        ay2 = min(max(ay1 + 2, iy2 + oy), h)
        crop = image_rgb[ay1:ay2, ax1:ax2]
        coords = detector.predict_crop(crop)  # crop-pixel space
        samples.append([[ax1 + float(cx), ay1 + float(cy)] for cx, cy in coords])

    abs_pts = np.median(np.asarray(samples, dtype=np.float64), axis=0).tolist()
    cej = [abs_pts[k] for k in (0, 1) if vis[k]]
    crest_line = [abs_pts[k] for k in (2, 3) if vis[k]]
    pred = {"tooth_id": tooth.get("tooth_id"), "cej": cej, "crest_line": crest_line}
    if vis[4]:
        pred["apex"] = [abs_pts[4]]
    return pred


def build_vitpose_model(pretrained: bool = True):
    """Load HF ViTPose and retarget its head to NUM_LANDMARKS heatmaps.

    The ViT backbone keeps its pose-pretrained weights; only the final 3x3 conv
    of the simple decoder is replaced (and randomly initialised) to emit K=5
    dental-landmark heatmaps instead of 17 COCO keypoints.
    """
    from torch import nn  # pragma: no cover
    from transformers import VitPoseConfig, VitPoseForPoseEstimation  # pragma: no cover

    if pretrained:  # pragma: no cover  (loads the 86M HF ViTPose checkpoint)
        model = VitPoseForPoseEstimation.from_pretrained(PRETRAINED_ID)
    else:  # pragma: no cover
        model = VitPoseForPoseEstimation(VitPoseConfig())
    in_ch = model.head.conv.in_channels  # pragma: no cover
    model.head.conv = nn.Conv2d(
        in_ch, NUM_LANDMARKS, kernel_size=3, stride=1, padding=1
    )  # pragma: no cover
    model.config.num_labels = NUM_LANDMARKS  # pragma: no cover
    return model  # pragma: no cover


def normalize_crop(crop_rgb):
    """HxWx3 uint8 crop -> (1, 3, INPUT_H, INPUT_W) normalised float tensor."""
    import numpy as np
    import torch

    img = crop_rgb.astype(np.float32) / 255.0
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    t = torch.nn.functional.interpolate(
        t, size=(INPUT_H, INPUT_W), mode="bilinear", align_corners=False
    )
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (t - mean) / std


def heatmaps_to_coords(heatmaps):
    """Sub-pixel argmax decode: (B, K, H, W) -> (B, K, 2) xy in heatmap pixels."""
    import torch

    B, K, H, W = heatmaps.shape
    idx = heatmaps.reshape(B, K, -1).argmax(dim=2)
    ys = (idx // W).float()
    xs = (idx % W).float()
    for b in range(B):
        for k in range(K):
            x, y = int(xs[b, k]), int(ys[b, k])
            if 1 <= x < W - 1:
                xs[b, k] += 0.25 * torch.sign(
                    heatmaps[b, k, y, x + 1] - heatmaps[b, k, y, x - 1]
                )
            if 1 <= y < H - 1:
                ys[b, k] += 0.25 * torch.sign(
                    heatmaps[b, k, y + 1, x] - heatmaps[b, k, y - 1, x]
                )
    return torch.stack([xs, ys], dim=2)


class ViTPoseLandmarkDetector:
    """Inference wrapper for the fine-tuned ViTPose dental detector."""

    def __init__(self, weights_path: str | Path, device: str = "auto") -> None:
        import torch

        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"ViTPose weights not found: {self.weights_path}")
        ckpt = torch.load(
            str(self.weights_path), map_location="cpu", weights_only=False
        )  # pragma: no cover
        self._model = build_vitpose_model(pretrained=False)  # pragma: no cover
        self._model.load_state_dict(ckpt["model_state_dict"])  # pragma: no cover
        if device == "auto":  # pragma: no cover
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)  # pragma: no cover
        self._model.to(self._device).eval()  # pragma: no cover
        self._torch = torch  # pragma: no cover

    def predict_crop(
        self, crop_rgb
    ) -> list[list[float]]:  # pragma: no cover  (needs the loaded HF model)
        """Predict K landmarks for one HxWx3 uint8 tooth crop.

        Returns K ``[x, y]`` coordinates in the crop's pixel space.
        """
        torch = self._torch
        h, w = crop_rgb.shape[:2]
        t = normalize_crop(crop_rgb).to(self._device)
        with torch.no_grad():
            hm = self._model(t).heatmaps
        coords = heatmaps_to_coords(hm)[0].cpu().numpy()
        out_h, out_w = hm.shape[2], hm.shape[3]
        sx, sy = w / out_w, h / out_h
        return [[float(x * sx), float(y * sy)] for x, y in coords]

    def predict_crop_conf(
        self, crop_rgb
    ):  # pragma: no cover  (needs the loaded HF model)
        """Like ``predict_crop`` but also returns a per-landmark confidence.

        Confidence is the spatial-softmax peak of each heatmap: a sharp unimodal
        heatmap gives a high peak; a flat or bimodal heatmap (the argmax-flip
        failure mode) gives a low peak. Returns ``(coords, confidences)``.
        """
        torch = self._torch
        h, w = crop_rgb.shape[:2]
        t = normalize_crop(crop_rgb).to(self._device)
        with torch.no_grad():
            hm = self._model(t).heatmaps
        coords = heatmaps_to_coords(hm)[0].cpu().numpy()
        out_h, out_w = hm.shape[2], hm.shape[3]
        flat = hm[0].reshape(hm.shape[1], -1)
        prob = torch.softmax(flat, dim=1)
        peak = prob.max(dim=1).values.cpu().numpy()
        sx, sy = w / out_w, h / out_h
        pts = [[float(x * sx), float(y * sy)] for x, y in coords]
        return pts, [float(p) for p in peak]
