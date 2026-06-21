#!/usr/bin/env python3
"""Convert DenPAR per-tooth landmark annotations into a YOLO-pose dataset.

Each tooth becomes one object (class 0) with 5 ordered keypoints
(cej_left, cej_right, crest_mesial, crest_distal, apex) and a bounding box enclosing
its visible landmarks. Writes outputs/denpar_pose/{images,labels}/{train,val,test} +
the dataset yaml for YOLO26-pose fine-tuning.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from toothprint.bench.data.denpar_adapter import RealDenparAdapter
from toothprint.bench.landmarks.vitpose_detector import tooth_to_landmarks

SRC = "data/denpar/extracted/Dataset"
OUT = Path("outputs/denpar_pose")
PAD = 0.35          # bbox padding as a fraction of the landmark span


def main():
    a = RealDenparAdapter(SRC)
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_img = n_tooth = 0
        for rec in a.records(split):
            with Image.open(rec.image_path) as im:
                W, H = im.size
            lines = []
            for tooth in rec.annotation_dict.get("teeth", []):
                pts, vis = tooth_to_landmarks(tooth)
                v = np.asarray(vis); p = np.asarray(pts, float)
                if v.sum() < 3:                         # need a few real landmarks
                    continue
                vp = p[v]
                x0, y0 = vp.min(0); x1, y1 = vp.max(0)
                px = PAD * max(x1 - x0, 8.0); py = PAD * max(y1 - y0, 8.0)
                bx0, by0, bx1, by1 = x0 - px, y0 - py, x1 + px, y1 + py
                cx = np.clip((bx0 + bx1) / 2 / W, 0, 1); cy = np.clip((by0 + by1) / 2 / H, 0, 1)
                ww = np.clip((bx1 - bx0) / W, 0, 1); hh = np.clip((by1 - by0) / H, 0, 1)
                kp = []
                for k in range(5):
                    if v[k]:
                        kp += [float(np.clip(p[k, 0] / W, 0, 1)), float(np.clip(p[k, 1] / H, 0, 1)), 2]
                    else:
                        kp += [0.0, 0.0, 0]
                lines.append("0 " + " ".join(f"{x:.6f}" for x in [cx, cy, ww, hh] + kp))
                n_tooth += 1
            if not lines:
                continue
            dst = OUT / "images" / split / rec.image_path.name
            if not dst.exists():
                os.symlink(rec.image_path.resolve(), dst)
            (OUT / "labels" / split / (rec.image_path.stem + ".txt")).write_text("\n".join(lines) + "\n")
            n_img += 1
        print(f"{split}: {n_img} images, {n_tooth} teeth", flush=True)

    (OUT / "denpar_pose.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "kpt_shape: [5, 3]\nflip_idx: [1, 0, 3, 2, 4]\nnames:\n  0: tooth\n")
    print("wrote", OUT / "denpar_pose.yaml", flush=True)


if __name__ == "__main__":
    main()
