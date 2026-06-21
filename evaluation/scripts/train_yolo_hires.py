#!/usr/bin/env python3
"""Fine-tune YOLO26-pose at near-native 1280 px (DenPAR is ~1168 px) for the sharpest
CEJ/bone-crest localisation — the lever that lifts end-to-end change recall toward the
0.98 measurement ceiling. Higher resolution gives finer keypoints than the 18 px median
of the 960 px model; the small backbone keeps it within an 8 GB GPU.

Run from the shared evaluation working dir (data/ + outputs/denpar_pose present).
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

DATA = Path("outputs/denpar_pose/denpar_pose.yaml").resolve()


def main():
    model = YOLO("yolo26s-pose.pt")
    model.train(
        data=str(DATA), epochs=140, imgsz=1280, batch=4, patience=35,
        project="outputs/yolo_pose", name="denpar26s_hi1280", exist_ok=True,
        deterministic=True, seed=0, close_mosaic=15, plots=True, verbose=True)
    print("best weights: outputs/yolo_pose/denpar26s_hi1280/weights/best.pt", flush=True)


if __name__ == "__main__":
    main()
