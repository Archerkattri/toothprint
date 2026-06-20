import unittest


def _annotation() -> dict:
    return {
        "image": "case001.png",
        "teeth": [
            {
                "tooth_id": "36",
                "cej": [[10.0, 20.0], [30.0, 20.0]],
                "apex": [[20.0, 80.0]],
                "crest_line": [[10.0, 35.0], [30.0, 35.0]],
            }
        ],
    }


class PipelineTests(unittest.TestCase):
    def test_evaluate_pairs_produces_decision_rows_from_conformal_interval(self):
        from dcc.benchmark.pipeline import evaluate_pairs
        from dcc.certificate.conformal import ConformalInterval
        from dcc.perturb.acquisition import TransformParams, apply_acquisition_perturbation
        from dcc.perturb.truechange import inject_crestal_change

        pairs = [
            apply_acquisition_perturbation(_annotation(), TransformParams(dy=2.0)),
            inject_crestal_change(_annotation(), tooth_id="36", delta_px=5.0),
        ]
        model = ConformalInterval(radius=0.5, alpha=0.1)

        rows = evaluate_pairs(pairs, tau=1.0, conformal=model, tooth_id="36")

        self.assertEqual(
            rows,
            [
                {"true": "stable", "decision": "stable", "score": 0.0, "lo": -0.5, "hi": 0.5},
                {"true": "progressed", "decision": "progressed", "score": 5.0, "lo": 4.5, "hi": 5.5},
            ],
        )


def test_score_uses_predicted_not_gt_when_store_provided():
    """Score path must consume predicted landmarks, not GT annotations."""
    from dcc.landmarks.store import PredictedLandmarkStore, LandmarkPrediction
    from dcc.benchmark.pipeline import evaluate_pairs
    from dcc.data.pair_builder import PairBuilderConfig, build_pairs

    # Build a minimal GT record
    class _Rec:
        image_id = "test_img"
        annotation_dict = {
            "image": "test_img",
            "teeth": [{"tooth_id": "1",
                        "cej": [[100.0, 200.0], [150.0, 200.0]],
                        "crest_line": [[100.0, 300.0], [150.0, 300.0]]}]
        }

    records = [_Rec()]
    cfg = PairBuilderConfig(acq_noise_std=0.0, crestal_shift_px=50.0, seed=0)
    pairs = build_pairs(records, cfg)

    # Create a predicted store where crest is shifted by a large known amount
    # so the predicted score differs significantly from the GT score
    big_shift_ann = {
        "image": "test_img",
        "teeth": [{"tooth_id": "1",
                    "cej": [[100.0, 200.0], [150.0, 200.0]],
                    "crest_line": [[100.0, 400.0], [150.0, 400.0]]}]  # crest 100px lower
    }
    store = PredictedLandmarkStore()
    store.add(LandmarkPrediction(image_id="test_img", annotation_dict=big_shift_ann, is_oracle=False))

    rows_predicted = evaluate_pairs(pairs, tau=10.0, landmark_store=store)
    rows_gt = evaluate_pairs(pairs, tau=10.0)

    # Predicted scores should differ from GT scores when the store has shifted landmarks
    pred_scores = [r.predicted_score for r in rows_predicted]
    gt_scores_plain = [r["score"] for r in rows_gt]
    assert pred_scores != gt_scores_plain, (
        "Score path appears to use GT landmarks even when landmark_store was provided"
    )


def test_oracle_store_raises_in_production_path():
    """evaluate_pairs must raise ValueError when an oracle store is passed."""
    import pytest
    from dcc.landmarks.store import PredictedLandmarkStore, LandmarkPrediction
    from dcc.benchmark.pipeline import evaluate_pairs
    from dcc.data.pair_builder import PairBuilderConfig, build_pairs

    class _Rec:
        image_id = "oracle_img"
        annotation_dict = {
            "image": "oracle_img",
            "teeth": [{"tooth_id": "1",
                        "cej": [[100.0, 200.0], [150.0, 200.0]],
                        "crest_line": [[100.0, 300.0], [150.0, 300.0]]}]
        }

    records = [_Rec()]
    cfg = PairBuilderConfig(acq_noise_std=0.0, crestal_shift_px=50.0, seed=0)
    pairs = build_pairs(records, cfg)

    oracle_store = PredictedLandmarkStore.from_gt_oracle(records)

    with pytest.raises(ValueError, match="Oracle prediction"):
        evaluate_pairs(pairs, tau=10.0, landmark_store=oracle_store)


_TEETH = [
    {
        "tooth_id": "1",
        "cej": [[100.0, 200.0], [150.0, 200.0]],
        "crest_line": [[100.0, 300.0], [150.0, 300.0]],
    }
]


def test_evaluate_with_store_raises_when_followup_is_oracle():
    """evaluate_pairs raises ValueError when the followup prediction has is_oracle=True (line 115)."""
    import pytest
    from dcc.landmarks.store import PredictedLandmarkStore, LandmarkPrediction
    from dcc.benchmark.pipeline import evaluate_pairs

    class _FakePair:
        baseline = {"image": "base_img", "teeth": _TEETH}
        followup = {"image": "foll_img", "teeth": _TEETH}
        label = "stable"

    store = PredictedLandmarkStore()
    store.add(LandmarkPrediction("base_img", annotation_dict={"teeth": _TEETH}, is_oracle=False))
    store.add(LandmarkPrediction("foll_img", annotation_dict={"teeth": _TEETH}, is_oracle=True))

    with pytest.raises(ValueError, match="Oracle prediction"):
        evaluate_pairs([_FakePair()], tau=10.0, landmark_store=store)


def test_evaluate_with_store_skips_pair_when_prediction_missing():
    """evaluate_pairs emits UserWarning and skips pairs with missing predictions (lines 122-128)."""
    import warnings
    from dcc.landmarks.store import PredictedLandmarkStore
    from dcc.benchmark.pipeline import evaluate_pairs

    class _FakePair:
        baseline = {"image": "img_not_in_store", "teeth": _TEETH}
        followup = {"image": "img_not_in_store", "teeth": _TEETH}
        label = "stable"

    store = PredictedLandmarkStore()  # empty — no predictions

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = evaluate_pairs([_FakePair()], tau=10.0, landmark_store=store)

    assert len(rows) == 0
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_evaluate_with_store_and_conformal_produces_bounded_interval():
    """evaluate_pairs uses conformal radius when store and conformal are both provided (lines 135-137)."""
    import pytest
    from dcc.landmarks.store import PredictedLandmarkStore, LandmarkPrediction
    from dcc.benchmark.pipeline import evaluate_pairs, EvalRow
    from dcc.certificate.conformal import ConformalInterval

    class _FakePair:
        baseline = {"image": "img1", "teeth": _TEETH}
        followup = {"image": "img1", "teeth": _TEETH}
        label = "stable"

    store = PredictedLandmarkStore()
    store.add(LandmarkPrediction("img1", annotation_dict={"teeth": _TEETH}, is_oracle=False))

    conformal = ConformalInterval(radius=5.0, alpha=0.1)
    rows = evaluate_pairs([_FakePair()], tau=20.0, conformal=conformal, landmark_store=store)

    assert len(rows) == 1
    assert isinstance(rows[0], EvalRow)
    assert rows[0].hi == pytest.approx(rows[0].predicted_score + 5.0)
    assert rows[0].lo == pytest.approx(rows[0].predicted_score - 5.0)


def test_evalrow_dict_style_access():
    """EvalRow supports __getitem__, __contains__, and .get so metrics
    helpers can consume store-path rows and dict rows uniformly."""
    from dcc.benchmark.pipeline import EvalRow

    row = EvalRow(
        true="stable", decision="stable", score=1.0, lo=0.5, hi=1.5,
        predicted_score=1.0, gt_score=0.9,
    )
    # __getitem__
    assert row["true"] == "stable"
    assert row["hi"] == 1.5
    # __contains__
    assert "lo" in row
    assert "nonexistent" not in row
    # .get
    assert row.get("decision") == "stable"
    assert row.get("missing", "default") == "default"


def test_store_lookup_falls_back_to_stem():
    """_store_lookup matches a full filename against a stem-keyed store."""
    from dcc.landmarks.store import PredictedLandmarkStore, LandmarkPrediction
    from dcc.benchmark.pipeline import _store_lookup

    store = PredictedLandmarkStore()
    store.add(LandmarkPrediction("Image28", annotation_dict={"teeth": _TEETH}, is_oracle=False))

    # None / empty ref → None
    assert _store_lookup(store, None) is None
    assert _store_lookup(store, "") is None
    # exact stem key hits directly
    assert _store_lookup(store, "Image28") is not None
    # full filename falls back to stem
    assert _store_lookup(store, "Image28.png") is not None
    # genuinely absent → None
    assert _store_lookup(store, "NoSuchImage.png") is None


if __name__ == "__main__":
    unittest.main()
