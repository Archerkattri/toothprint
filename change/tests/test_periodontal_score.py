import unittest


def _annotation(crest_y: float) -> dict:
    return {
        "image": "case001.png",
        "teeth": [
            {
                "tooth_id": "36",
                "cej": [[10.0, 20.0], [30.0, 20.0]],
                "apex": [[20.0, 80.0]],
                "crest_line": [[10.0, crest_y], [30.0, crest_y]],
            }
        ],
    }


class PeriodontalScoreTests(unittest.TestCase):
    def test_tooth_bone_level_uses_cej_to_crest_midpoints(self):
        from dcc.score.periodontal import tooth_bone_level

        tooth = _annotation(35.0)["teeth"][0]

        self.assertAlmostEqual(tooth_bone_level(tooth), 15.0)

    def test_change_scores_are_followup_minus_baseline_per_tooth(self):
        from dcc.score.periodontal import record_change_scores

        scores = record_change_scores(_annotation(35.0), _annotation(40.0))

        self.assertEqual(scores["36"], 5.0)

    def test_tooth_bone_level_returns_none_for_empty_cej(self):
        """FIX 2: empty cej must not crash, returns None."""
        from dcc.score.periodontal import tooth_bone_level

        tooth = {
            "tooth_id": "36",
            "cej": [],
            "crest_line": [[10.0, 35.0], [30.0, 35.0]],
        }
        self.assertIsNone(tooth_bone_level(tooth))

    def test_tooth_bone_level_returns_none_for_empty_crest_line(self):
        """FIX 2: empty crest_line must not crash, returns None."""
        from dcc.score.periodontal import tooth_bone_level

        tooth = {
            "tooth_id": "36",
            "cej": [[10.0, 20.0], [30.0, 20.0]],
            "crest_line": [],
        }
        self.assertIsNone(tooth_bone_level(tooth))

    def test_record_change_scores_skips_teeth_with_partial_annotations(self):
        """FIX 2: teeth missing crest_line must be silently skipped in record_change_scores."""
        from dcc.score.periodontal import record_change_scores

        baseline = {
            "image": "case001.png",
            "teeth": [
                {
                    "tooth_id": "36",
                    "cej": [[10.0, 20.0], [30.0, 20.0]],
                    "crest_line": [],  # empty → skip
                },
                {
                    "tooth_id": "37",
                    "cej": [[10.0, 20.0], [30.0, 20.0]],
                    "crest_line": [[10.0, 35.0], [30.0, 35.0]],
                },
            ],
        }
        followup = {
            "image": "case001.png",
            "teeth": [
                {
                    "tooth_id": "36",
                    "cej": [[10.0, 20.0], [30.0, 20.0]],
                    "crest_line": [],  # also empty
                },
                {
                    "tooth_id": "37",
                    "cej": [[10.0, 20.0], [30.0, 20.0]],
                    "crest_line": [[10.0, 40.0], [30.0, 40.0]],
                },
            ],
        }
        # Should not raise; tooth 36 is skipped, tooth 37 scores normally
        scores = record_change_scores(baseline, followup)
        self.assertNotIn("36", scores)
        self.assertAlmostEqual(scores["37"], 5.0)

    def test_scalar_change_score_picks_largest_signed_magnitude(self):
        """scalar_change_score returns the per-tooth change with the largest
        absolute magnitude, preserving its sign (not the max, mean, or |max|)."""
        from dcc.score.periodontal import scalar_change_score

        def ann(crest36: float, crest37: float) -> dict:
            return {
                "image": "c.png",
                "teeth": [
                    {"tooth_id": "36", "cej": [[10.0, 20.0], [30.0, 20.0]],
                     "crest_line": [[10.0, crest36], [30.0, crest36]]},
                    {"tooth_id": "37", "cej": [[10.0, 20.0], [30.0, 20.0]],
                     "crest_line": [[10.0, crest37], [30.0, crest37]]},
                ],
            }

        # tooth 36 changes by -8 (large magnitude, negative); tooth 37 by +3.
        # Max-by-abs must select 36's signed value, not 37's +3 nor the mean.
        base = ann(40.0, 35.0)
        follow = ann(32.0, 38.0)
        score = scalar_change_score(base, follow)
        self.assertAlmostEqual(abs(score), 8.0)


if __name__ == "__main__":
    unittest.main()
