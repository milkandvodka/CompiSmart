import unittest
from unittest.mock import patch

from comparag.creative_features import analyze_creative_features, prepare_creative_evidence


class CreativeFeaturesTests(unittest.TestCase):
    def test_evidence_mode_prepares_transcript_without_semantic_inference(self):
        video = {
            "transcript": {
                "variants": {
                    "english_normalized": (
                        "If you use this workflow every morning, reports get cleaner. "
                        "It includes a checklist and review step. You will see fewer mistakes."
                    )
                },
                "segments": [],
            }
        }

        result = prepare_creative_evidence(video, video_id="B")

        self.assertTrue(result["evidence_available"])
        self.assertIn("First 5 seconds transcript", result["evidence_pack"])
        self.assertIsNone(result["hook_type"])
        self.assertEqual(result["pain_points"], [])
        self.assertEqual(result["proof_elements"], [])
        self.assertEqual(result["risk_flags"], [])

    @patch("comparag.creative_features.gemini_json")
    def test_llm_creative_features_populates_semantic_fields(self, gemini_mock):
        gemini_mock.return_value = {
            "hook_type": "problem_solution",
            "first_5s_promise": "Reports get cleaner.",
            "target_audience": "busy operators",
            "pain_points": ["messy reports"],
            "proof_elements": ["checklist"],
            "cta": "try the workflow",
            "emotional_angle": "relief",
            "claims": [],
            "risk_flags": [],
            "improvement_opportunities": ["Add stronger CTA."],
            "notes": [],
        }
        video = {"transcript": {"text": "Reports get cleaner with a checklist."}}

        result = analyze_creative_features(video, video_id="B", mode="llm", model="test-model")

        self.assertTrue(result["llm_used"])
        self.assertEqual(result["hook_type"], "problem_solution")
        self.assertIn("transcript-only creative features", gemini_mock.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
