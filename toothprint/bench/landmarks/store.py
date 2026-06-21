"""Predicted landmark store: holds detector output per image, separate from GT.

The score path MUST consume predicted landmarks, never GT annotations.
This module enforces that separation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class LandmarkPrediction:
    """Predicted landmarks for one image with optional confidence scores."""
    image_id: str
    annotation_dict: dict   # DenPAR-style: {"image": ..., "teeth": [...]}
    is_oracle: bool = False  # True only for GT (oracle) predictions


class PredictedLandmarkStore:
    """Stores predicted (or oracle) landmarks keyed by image_id.

    For validation/research, supports side-by-side oracle vs predicted comparison.
    In production use, is_oracle must be False for all entries.
    """

    def __init__(self) -> None:
        self._predictions: dict[str, LandmarkPrediction] = {}

    def add(self, pred: LandmarkPrediction) -> None:
        self._predictions[pred.image_id] = pred

    def get(self, image_id: str) -> Optional[LandmarkPrediction]:
        return self._predictions.get(image_id)

    def __len__(self) -> int:
        return len(self._predictions)

    @classmethod
    def from_vitpose(cls, gt_records: list, detector) -> "PredictedLandmarkStore":
        """Build a store from REAL ViTPose detections, one crop per GT tooth.

        Top-down inference: each GT tooth's landmarks define a square crop
        (``landmark_box``); ``detector.predict_crop`` returns the 5 landmark
        coordinates in crop-pixel space, which are mapped back to image pixels.
        The predicted tooth mirrors the GT tooth's visible-landmark structure
        (same ``tooth_id``, same cej/crest_line/apex fields) but with detector
        coordinates — never the GT coordinates. is_oracle=False.

        Tooth localisation comes from the GT box, so the resulting error is the
        landmark-detection error in isolation (matching the train/val px metric),
        not localisation error.

        ``detector`` is any object exposing ``predict_crop(crop_rgb) ->
        [[x, y], ...]`` (a ``ViTPoseLandmarkDetector``); it is injected so the
        store has no hard dependency on the HF model weights.
        """
        from PIL import Image as _Image
        from toothprint.bench.landmarks.vitpose_detector import predict_tooth

        store = cls()
        for rec in gt_records:
            img = np.array(_Image.open(rec.image_path).convert("RGB"))
            pred_teeth = []
            for tooth in rec.annotation_dict.get("teeth", []):
                pred_tooth = predict_tooth(detector, img, tooth)
                if pred_tooth is not None:
                    pred_teeth.append(pred_tooth)
            store.add(LandmarkPrediction(
                image_id=rec.image_id,
                annotation_dict={"teeth": pred_teeth},
                is_oracle=False,
            ))
        return store

    @classmethod
    def from_gt_oracle(cls, gt_records: list) -> "PredictedLandmarkStore":
        """Build a store from GT annotations with is_oracle=True.

        Use ONLY for oracle comparison in research — never in production scoring.
        """
        store = cls()
        for rec in gt_records:
            ann = rec.annotation_dict
            store.add(LandmarkPrediction(
                image_id=rec.image_id,
                annotation_dict=ann,
                is_oracle=True,
            ))
        return store


    def for_image(self, image_id: str) -> "LandmarkPrediction":
        """Alias for get() that raises KeyError on miss (legacy compat)."""
        result = self.get(image_id)
        if result is None:
            raise KeyError(f"No predicted landmarks for image_id={image_id!r}")
        return result

    @classmethod
    def load(cls, path: "Path | str") -> "PredictedLandmarkStore":
        """Load predictions from JSON file written by to_json()."""
        import json
        from pathlib import Path as _Path
        payload = json.loads(_Path(path).read_text(encoding="utf-8"))
        store = cls()
        for record in payload.get("records", []):
            image_id = record.get("image_id")
            if not image_id:
                raise ValueError("Every landmark prediction record needs image_id")
            store.add(LandmarkPrediction(
                image_id=image_id,
                annotation_dict={"teeth": record.get("teeth", [])},
                is_oracle=False,
            ))
        return store

    def to_json(self, path: "Path | str") -> "Path":
        """Serialize store to JSON (paired with load())."""
        import json
        from pathlib import Path as _Path
        output = _Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": [
                {"image_id": image_id, "teeth": pred.annotation_dict.get("teeth", [])}
                for image_id, pred in sorted(self._predictions.items())
            ]
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output
