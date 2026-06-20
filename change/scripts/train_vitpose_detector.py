#!/usr/bin/env python3
"""Fine-tune the pose-pretrained ViTPose (HF) on DenPAR dental landmarks.

Top-down heatmap regression — the state-of-the-art replacement for the dated
KeypointRCNN keypoint head. Each tooth is cropped to ViTPose's 256x192 input,
the network predicts 5 Gaussian heatmaps (cej_left, cej_right, crest_mesial,
crest_distal, apex), and training minimises masked heatmap MSE. Validation
reports the per-landmark pixel error in ORIGINAL image space, directly
comparable to the KeypointRCNN baseline.

Usage:
    python scripts/train_vitpose_detector.py \
        --data data/denpar/extracted/Dataset \
        --output outputs/vitpose_detector \
        --epochs 20 --batch-size 16 --lr 5e-4 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from dcc.data.denpar_adapter import RealDenparAdapter
from dcc.landmarks.vitpose_detector import (
    INPUT_H,
    INPUT_W,
    LANDMARK_NAMES,
    NUM_LANDMARKS,
    build_vitpose_model,
)

HEATMAP_H, HEATMAP_W = INPUT_H // 4, INPUT_W // 4  # ViTPose decoder upsamples x4
_SIGMA = 2.0


# Top-down crop helpers are shared with inference (store.from_vitpose) so the
# training and deployment framing stay identical.
from dcc.landmarks.vitpose_detector import (  # noqa: E402
    landmark_box as _square_box,
    tooth_to_landmarks as _tooth_landmarks,
)


def _gaussian_heatmap(cx: float, cy: float):
    """A single (HEATMAP_H, HEATMAP_W) Gaussian centred at (cx, cy) in heatmap px."""
    xs = np.arange(HEATMAP_W)[None, :]
    ys = np.arange(HEATMAP_H)[:, None]
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * _SIGMA**2)).astype(np.float32)


def _build_instances(records):
    """Yield (image_path, box, pts[5,2], vis[5]) tooth instances."""
    instances = []
    for rec in records:
        for tooth in rec.annotation_dict.get("teeth", []):
            pts, vis = _tooth_landmarks(tooth)
            if not vis.any():
                continue
            instances.append((rec.image_path, pts, vis))
    return instances


class _DenparViTPoseDataset:
    def __init__(self, instances, torch):
        self.instances = instances
        self.torch = torch
        from PIL import Image
        self._Image = Image

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        import torch
        from dcc.landmarks.vitpose_detector import normalize_crop

        img_path, pts, vis = self.instances[i]
        img = np.array(self._Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]
        x1, y1, x2, y2 = _square_box(pts, vis, w, h)
        crop = img[int(y1):int(y2), int(x1):int(x2)]
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            crop = img
            x1 = y1 = 0.0
            x2, y2 = float(w), float(h)
        ch, cw = crop.shape[:2]
        pixel = normalize_crop(crop)[0]  # (3, INPUT_H, INPUT_W)

        # Targets: landmark -> crop-normalised -> heatmap pixel.
        target = np.zeros((NUM_LANDMARKS, HEATMAP_H, HEATMAP_W), dtype=np.float32)
        weight = np.zeros((NUM_LANDMARKS,), dtype=np.float32)
        for k in range(NUM_LANDMARKS):
            if not vis[k]:
                continue
            nx = (pts[k, 0] - x1) / max(cw, 1) * HEATMAP_W
            ny = (pts[k, 1] - y1) / max(ch, 1) * HEATMAP_H
            if 0 <= nx < HEATMAP_W and 0 <= ny < HEATMAP_H:
                target[k] = _gaussian_heatmap(nx, ny)
                weight[k] = 1.0
        meta = np.array([x1, y1, x2, y2], dtype=np.float32)
        return (
            pixel,
            torch.from_numpy(target),
            torch.from_numpy(weight),
            torch.from_numpy(pts),
            torch.from_numpy(vis.astype(np.float32)),
            torch.from_numpy(meta),
        )


def _decode_px_error(heatmaps, pts, vis, meta, torch):
    """Per-landmark pixel error in ORIGINAL image space for a batch."""
    from dcc.landmarks.vitpose_detector import heatmaps_to_coords

    coords = heatmaps_to_coords(heatmaps)  # (B, K, 2) heatmap px
    B = heatmaps.shape[0]
    errs = {name: [] for name in LANDMARK_NAMES}
    for b in range(B):
        x1, y1, x2, y2 = meta[b].tolist()
        cw, ch = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        for k in range(NUM_LANDMARKS):
            if vis[b, k] < 0.5:
                continue
            px = x1 + float(coords[b, k, 0]) / HEATMAP_W * cw
            py = y1 + float(coords[b, k, 1]) / HEATMAP_H * ch
            gx, gy = float(pts[b, k, 0]), float(pts[b, k, 1])
            errs[LANDMARK_NAMES[k]].append(((px - gx) ** 2 + (py - gy) ** 2) ** 0.5)
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune ViTPose on DenPAR landmarks")
    ap.add_argument("--data", default="data/denpar/extracted/Dataset")
    ap.add_argument("--output", default="outputs/vitpose_detector")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1,
                    help="LR multiplier for the pretrained backbone")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    adapter = RealDenparAdapter(args.data)
    train_recs = list(adapter.records(split="train"))
    val_recs = list(adapter.records(split="val"))
    train_inst = _build_instances(train_recs)
    val_inst = _build_instances(val_recs)
    print(f"Loaded {len(train_recs)}/{len(val_recs)} images -> "
          f"{len(train_inst)}/{len(val_inst)} tooth instances")

    train_ds = _DenparViTPoseDataset(train_inst, torch)
    val_ds = _DenparViTPoseDataset(val_inst, torch)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_vitpose_model(pretrained=True).to(device)
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    best_px = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for pixel, target, weight, _pts, _vis, _meta in train_dl:
            pixel, target, weight = pixel.to(device), target.to(device), weight.to(device)
            hm = model(pixel).heatmaps
            # Masked per-keypoint MSE.
            per_kp = ((hm - target) ** 2).mean(dim=(2, 3))  # (B, K)
            loss = (per_kp * weight).sum() / weight.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss) * pixel.shape[0]
        sched.step()
        train_loss = running / max(len(train_ds), 1)

        # ---- validation: per-landmark pixel error ----
        model.eval()
        all_errs = {name: [] for name in LANDMARK_NAMES}
        with torch.no_grad():
            for pixel, _t, _w, pts, vis, meta in val_dl:
                hm = model(pixel.to(device)).heatmaps.cpu()
                batch = _decode_px_error(hm, pts, vis, meta, torch)
                for name in LANDMARK_NAMES:
                    all_errs[name].extend(batch[name])
        per_lm = {name: (float(np.mean(v)) if v else float("nan")) for name, v in all_errs.items()}
        overall = float(np.mean([e for v in all_errs.values() for e in v]))
        log.append({"epoch": epoch, "train_loss": train_loss, "val_overall_px_error": overall, "per_landmark_px": per_lm})
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.5f}  val_overall_px_error={overall:.3f}")
        for name in LANDMARK_NAMES:
            print(f"    {name:<14} mean_px={per_lm[name]:.3f}")

        if overall < best_px:
            best_px = overall
            torch.save(
                {"model_state_dict": model.state_dict(), "val_overall_px_error": overall,
                 "landmark_names": list(LANDMARK_NAMES), "arch": "vitpose-base-simple"},
                out_dir / "checkpoint_best.pt",
            )
            print(f"    new best checkpoint (val_overall_px_error={overall:.4f})")

    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(f"\nDone. Best val overall px error: {best_px:.4f}")
    print(f"Checkpoints + log written to {out_dir}")


if __name__ == "__main__":
    main()
