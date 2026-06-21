#!/usr/bin/env python3
"""Keypoint-localisation error (px) of the trained YOLO26-pose detector on the DenPAR
test split, vs the ViTPose ~38 px baseline. Predicted teeth are matched to ground-truth
teeth by box centre; per visible landmark the pixel error is in original-image coords.
Reports mean/median/p90 px, PRCK, and the tooth-detection rate. Writes a JSON + figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from toothprint.bench.data.denpar_adapter import RealDenparAdapter
from toothprint.bench.landmarks.vitpose_detector import LANDMARK_NAMES, tooth_to_landmarks

SRC = "data/denpar/extracted/Dataset"
WEIGHTS = "runs/pose/outputs/yolo_pose/denpar26s/weights/best.pt"
VITPOSE_PX = 38.0


def gt_teeth(rec):
    out = []
    for tooth in rec.annotation_dict.get("teeth", []):
        pts, vis = tooth_to_landmarks(tooth)
        v = np.asarray(vis); p = np.asarray(pts, float)
        if v.sum() >= 3:
            out.append((p, v))
    return out


def main():
    model = YOLO(WEIGHTS)
    a = RealDenparAdapter(SRC)
    err, err_by_k = [], {k: [] for k in range(5)}
    n_gt = n_matched = 0
    for rec in a.records("test"):
        gts = gt_teeth(rec); n_gt += len(gts)
        if not gts:
            continue
        res = model(str(rec.image_path), imgsz=960, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            continue
        pk = res.keypoints.xy.cpu().numpy()             # (M,5,2) original px
        pc = res.boxes.xywh.cpu().numpy()[:, :2]        # (M,2) box centres
        used = set()
        for p_gt, v_gt in gts:
            gc = p_gt[v_gt].mean(0)
            order = np.argsort(((pc - gc) ** 2).sum(1))
            j = next((int(o) for o in order if int(o) not in used), None)
            span = np.linalg.norm(p_gt[v_gt].max(0) - p_gt[v_gt].min(0))
            if j is None or np.linalg.norm(pc[j] - gc) > span + 60:
                continue                                # detector missed this tooth
            used.add(j); n_matched += 1
            for k in range(5):
                if v_gt[k]:
                    e = float(np.linalg.norm(pk[j, k] - p_gt[k]))
                    err.append(e); err_by_k[k].append(e)
    err = np.array(err)
    # The change pipeline localizes a tooth from its CEJ + bone-crest only (cej_c, crest_c);
    # the apex is annotated but unused, and is the hardest landmark (deep in bone). Report the
    # used-landmark error separately — that is what caps end-to-end change recall.
    USED = (0, 1, 2, 3)                                       # cej_left/right, crest_mesial/distal
    err_used = np.array([e for k in USED for e in err_by_k[k]])
    res = {
        "weights": WEIGHTS, "n_gt_teeth": n_gt, "n_matched": n_matched,
        "detection_rate": round(n_matched / max(n_gt, 1), 3),
        "px_median_used": round(float(np.median(err_used)), 2),
        "px_mean_used": round(float(err_used.mean()), 2),
        "px_p90_used": round(float(np.percentile(err_used, 90)), 2),
        "prck_used": {f"{K}px": round(float((err_used < K).mean()), 3) for K in (5, 10, 20, 38)},
        "px_median_all5": round(float(np.median(err)), 2),
        "px_mean_all5": round(float(err.mean()), 2),
        "per_landmark_median_px": {LANDMARK_NAMES[k]: round(float(np.median(err_by_k[k])), 2)
                                   for k in range(5) if err_by_k[k]},
        "vitpose_baseline_px": VITPOSE_PX,
    }
    Path("outputs/yolo_pose").mkdir(parents=True, exist_ok=True)
    Path("outputs/yolo_pose/px_eval.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    med_used = float(np.median(err_used))
    ax[0].hist(err_used, bins=np.linspace(0, 60, 50), color="#2c6fb0", alpha=0.85)
    ax[0].axvline(med_used, color="#1a7f4b", lw=2, label=f"YOLO26 median {med_used:.1f}px")
    ax[0].axvline(VITPOSE_PX, color="#c0392b", lw=2, ls="--", label=f"ViTPose ~{VITPOSE_PX:.0f}px")
    ax[0].set_xlabel("CEJ/bone-crest localisation error (px)"); ax[0].set_ylabel("landmarks"); ax[0].legend(fontsize=9)
    ax[0].set_title(f"DenPAR test · {n_matched}/{n_gt} teeth detected", fontsize=11)
    Ks = [2, 5, 10, 20, 38]
    ax[1].plot(Ks, [(err_used < K).mean() for K in Ks], "-o", color="#2c6fb0")
    ax[1].axvline(VITPOSE_PX, color="#c0392b", ls="--", lw=1.5)
    ax[1].set_xlabel("threshold (px)"); ax[1].set_ylabel("PRCK (fraction within threshold)")
    ax[1].set_ylim(0, 1.02); ax[1].set_title("PRCK — precise to a few px, vs ~38px ViTPose", fontsize=11)
    fig.suptitle("YOLO26-pose CEJ/bone-crest localisation on DenPAR (apex excluded — unused by change)",
                 fontsize=12.5, y=1.0)
    fig.patch.set_facecolor("white"); fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("docs/detector_px.png", dpi=120); plt.close(fig)
    print("wrote docs/detector_px.png")


if __name__ == "__main__":
    main()
