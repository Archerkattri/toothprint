#!/usr/bin/env python3
"""Fine-tune YOLO26-pose at near-native 1280 px (DenPAR is ~1168 px) to test whether
higher-resolution training sharpens CEJ/bone-crest localisation below the 960 px model's
18 px median — the lever that would lift end-to-end change recall toward the 0.98 ceiling.

NEGATIVE RESULT (kept on purpose, so it is not re-run): 1280 px did NOT help — CEJ/crest
median 20.1 px (vs 18.2 px at 960 px) and end-to-end recall 0.87 (vs 0.905). The ~18 px
floor is the DenPAR annotation-label noise, not model capacity, so a bigger/finer detector
cannot close it; only better labels or real longitudinal pairs can. The 960 px model
(train_yolo26_pose.py) remains the detector of record.

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
