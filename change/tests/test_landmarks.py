"""Tests for the predicted-landmark store and ViTPose-driven store builder."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest


class PredictedLandmarkStoreTests(unittest.TestCase):
    def test_prediction_store_loads_predicted_landmarks_by_image_id(self):
        from dcc.landmarks.store import PredictedLandmarkStore

        payload = {
            "records": [
                {
                    "image_id": "case001",
                    "teeth": [
                        {
                            "tooth_id": "36",
                            "cej": [[10.0, 20.0], [30.0, 20.0]],
                            "apex": [[20.0, 80.0]],
                            "crest_line": [[10.0, 35.0], [30.0, 35.0]],
                        }
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            store = PredictedLandmarkStore.load(path)

        prediction = store.for_image("case001")

        self.assertEqual(prediction.annotation_dict["teeth"][0]["tooth_id"], "36")
        self.assertEqual(prediction.annotation_dict["teeth"][0]["crest_line"][0], [10.0, 35.0])

    def test_prediction_store_rejects_unknown_image_id(self):
        from dcc.landmarks.store import PredictedLandmarkStore, LandmarkPrediction

        store = PredictedLandmarkStore()
        store.add(LandmarkPrediction(
            image_id="case001",
            annotation_dict={"teeth": []},
            is_oracle=False,
        ))

        with self.assertRaisesRegex(KeyError, "case404"):
            store.for_image("case404")

    def test_prediction_store_to_json_round_trips(self):
        from dcc.landmarks.store import PredictedLandmarkStore, LandmarkPrediction

        store = PredictedLandmarkStore()
        store.add(LandmarkPrediction(
            image_id="img_a",
            annotation_dict={"teeth": [{"tooth_id": "1", "cej": [[0.0, 0.0]], "crest_line": [], "apex": []}]},
            is_oracle=False,
        ))
        store.add(LandmarkPrediction(
            image_id="img_b",
            annotation_dict={"teeth": []},
            is_oracle=False,
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.json"
            store.to_json(out)
            reloaded = PredictedLandmarkStore.load(out)

        self.assertEqual(reloaded.for_image("img_a").annotation_dict["teeth"][0]["tooth_id"], "1")
        self.assertEqual(reloaded.for_image("img_b").annotation_dict["teeth"], [])

    def test_prediction_store_raises_on_missing_image_id_in_payload(self):
        from dcc.landmarks.store import PredictedLandmarkStore

        payload = {"records": [{"teeth": []}]}  # no image_id field
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                PredictedLandmarkStore.load(path)

    def test_len_returns_count_of_predictions(self):
        from dcc.landmarks.store import PredictedLandmarkStore, LandmarkPrediction

        store = PredictedLandmarkStore()
        self.assertEqual(len(store), 0)
        store.add(LandmarkPrediction("img1", annotation_dict={"teeth": []}, is_oracle=False))
        self.assertEqual(len(store), 1)
        store.add(LandmarkPrediction("img2", annotation_dict={"teeth": []}, is_oracle=False))
        self.assertEqual(len(store), 2)

    def test_from_gt_oracle_marks_predictions_oracle(self):
        from dcc.landmarks.store import PredictedLandmarkStore

        class _GtRec:
            image_id = "gt_img"
            annotation_dict = {"teeth": [{"tooth_id": "1", "cej": [[1.0, 2.0]]}]}

        store = PredictedLandmarkStore.from_gt_oracle([_GtRec()])
        self.assertEqual(len(store), 1)
        self.assertTrue(store.get("gt_img").is_oracle)


class _FakeViTPose:
    """Stand-in detector: returns a fixed set of 5 crop-space landmarks."""

    def __init__(self, coords):
        self._coords = coords
        self.calls = 0

    def predict_crop(self, crop_rgb):
        self.calls += 1
        return [list(c) for c in self._coords]


class _Rec:
    def __init__(self, image_path, image_id, teeth):
        self.image_path = image_path
        self.image_id = image_id
        self.annotation_dict = {"teeth": teeth}


def _write_image(path, w, h):
    from PIL import Image
    Image.new("RGB", (w, h), color=(120, 120, 120)).save(path)


class VitposeStoreTests(unittest.TestCase):
    def test_from_vitpose_maps_crop_coords_back_to_image(self):
        from dcc.landmarks.store import PredictedLandmarkStore

        coords = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        detector = _FakeViTPose(coords)

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "x.png"
            _write_image(img_path, 200, 200)
            tooth = {
                "tooth_id": "11",
                "cej": [[80.0, 90.0], [110.0, 90.0]],
                "crest_line": [[80.0, 120.0], [110.0, 120.0]],
                "apex": [[95.0, 160.0]],
            }
            rec = _Rec(img_path, "x", [tooth])
            store = PredictedLandmarkStore.from_vitpose([rec], detector)

        self.assertEqual(len(store), 1)
        pred = store.get("x")
        self.assertFalse(pred.is_oracle)
        pt = pred.annotation_dict["teeth"][0]
        self.assertEqual(pt["tooth_id"], "11")
        self.assertEqual(detector.calls, 1)
        # box origin: landmarks span x[80,110] y[90,160]; margin=0.45*70+15=46.5
        # x1=max(0,80-46.5)=33 -> int 33 ; y1=max(0,90-46.5)=43 -> int 43
        self.assertEqual(pt["cej"][0], [33 + 1.0, 43 + 2.0])
        self.assertEqual(pt["cej"][1], [33 + 3.0, 43 + 4.0])
        self.assertEqual(pt["crest_line"][0], [33 + 5.0, 43 + 6.0])
        self.assertEqual(pt["crest_line"][1], [33 + 7.0, 43 + 8.0])
        self.assertEqual(pt["apex"], [[33 + 9.0, 43 + 10.0]])

    def test_from_vitpose_skips_tooth_without_visible_landmarks(self):
        from dcc.landmarks.store import PredictedLandmarkStore

        detector = _FakeViTPose([[0, 0]] * 5)
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "x.png"
            _write_image(img_path, 50, 50)
            empty_tooth = {"tooth_id": "1", "cej": [], "crest_line": [], "apex": []}
            rec = _Rec(img_path, "x", [empty_tooth])
            store = PredictedLandmarkStore.from_vitpose([rec], detector)

        self.assertEqual(store.get("x").annotation_dict["teeth"], [])
        self.assertEqual(detector.calls, 0)

    def test_from_vitpose_omits_apex_when_gt_has_none(self):
        from dcc.landmarks.store import PredictedLandmarkStore

        detector = _FakeViTPose([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "x.png"
            _write_image(img_path, 200, 200)
            tooth = {
                "tooth_id": "21",
                "cej": [[80.0, 90.0], [110.0, 90.0]],
                "crest_line": [[80.0, 120.0], [110.0, 120.0]],
                "apex": [],
            }
            rec = _Rec(img_path, "x", [tooth])
            store = PredictedLandmarkStore.from_vitpose([rec], detector)

        pt = store.get("x").annotation_dict["teeth"][0]
        self.assertNotIn("apex", pt)
        self.assertEqual(len(pt["cej"]), 2)

    def test_predict_tooth_tta_returns_median(self):
        from dcc.landmarks.vitpose_detector import predict_tooth

        # Detector returns crop-centre-ish coords; with n_tta the box jitters and
        # the per-landmark median is returned. Just assert a well-formed result.
        coords = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]
        detector = _FakeViTPose(coords)
        img = np.full((200, 200, 3), 120, dtype=np.uint8)
        tooth = {
            "tooth_id": "11",
            "cej": [[80.0, 90.0], [110.0, 90.0]],
            "crest_line": [[80.0, 120.0], [110.0, 120.0]],
            "apex": [[95.0, 160.0]],
        }
        pred = predict_tooth(detector, img, tooth, n_tta=5, jitter_px=3,
                             rng=np.random.default_rng(0))
        self.assertEqual(pred["tooth_id"], "11")
        self.assertEqual(len(pred["cej"]), 2)
        self.assertEqual(len(pred["crest_line"]), 2)
        self.assertIn("apex", pred)
        self.assertGreater(detector.calls, 1)  # multiple TTA passes ran

    def test_predict_tooth_degenerate_box_returns_none(self):
        from dcc.landmarks.vitpose_detector import predict_tooth

        detector = _FakeViTPose([[0, 0]] * 5)
        img = np.zeros((40, 1, 3), dtype=np.uint8)  # width 1 -> box < 2px
        tooth = {"tooth_id": "1", "cej": [[0.0, 10.0], [0.0, 12.0]],
                 "crest_line": [[0.0, 20.0], [0.0, 22.0]]}
        self.assertIsNone(predict_tooth(detector, img, tooth, n_tta=4))
        self.assertEqual(detector.calls, 0)

    def test_from_vitpose_skips_degenerate_crop(self):
        from dcc.landmarks.store import PredictedLandmarkStore

        detector = _FakeViTPose([[0, 0]] * 5)
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "x.png"
            _write_image(img_path, 1, 40)  # width 1 -> crop width < 2
            tooth = {
                "tooth_id": "1",
                "cej": [[0.0, 10.0], [0.0, 12.0]],
                "crest_line": [[0.0, 20.0], [0.0, 22.0]],
                "apex": [[0.0, 30.0]],
            }
            rec = _Rec(img_path, "x", [tooth])
            store = PredictedLandmarkStore.from_vitpose([rec], detector)

        self.assertEqual(store.get("x").annotation_dict["teeth"], [])
        self.assertEqual(detector.calls, 0)


if __name__ == "__main__":
    unittest.main()
