#!/usr/bin/env python3
"""Fine-tune YOLO26-pose (COCO-pretrained) on DenPAR for precise CEJ/bone-crest
localisation — replacing the ~36 px ViTPose detector that caps end-to-end change recall.
High input resolution (the radiographs are ~1168 px) for sub-pixel-ish keypoints.
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

DATA = Path("outputs/denpar_pose/denpar_pose.yaml").resolve()


def main():
    model = YOLO("yolo26s-pose.pt")
    model.train(
        data=str(DATA), epochs=120, imgsz=960, batch=8, patience=30,
        project="outputs/yolo_pose", name="denpar26s", exist_ok=True,
        deterministic=True, seed=0, close_mosaic=15, plots=True, verbose=True)
    print("best weights: outputs/yolo_pose/denpar26s/weights/best.pt", flush=True)


if __name__ == "__main__":
    main()
