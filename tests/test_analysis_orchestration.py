import unittest

from comparag.analysis import AnalysisConfig, enrich_payload_with_analysis
from comparag.metrics import build_video_profiles


class AnalysisOrchestrationTests(unittest.TestCase):
    def test_enrich_payload_attaches_cached_analysis_when_fingerprints_match(self):
        payload = {
            "videos": [
                {
                    "platform": "youtube",
                    "id": "yt",
                    "public_comment_objects": [{"id": "1", "text": "Nice", "like_count": 2}],
                    "transcript": {"text": "Nice hook."},
                }
            ]
        }
        profiles = build_video_profiles(payload, "demo")
        enriched, artifacts = enrich_payload_with_analysis(
            payload,
            profiles=profiles,
            previous_record=None,
            config=AnalysisConfig(comment_intelligence="evidence", creative_features="evidence"),
        )
        previous_record = {
            "comment_intelligence": artifacts["comment_intelligence"],
            "creative_features": artifacts["creative_features"],
        }

        enriched_again, artifacts_again = enrich_payload_with_analysis(
            payload,
            profiles=profiles,
            previous_record=previous_record,
            config=AnalysisConfig(comment_intelligence="evidence", creative_features="evidence"),
        )

        self.assertTrue(artifacts_again["comment_intelligence"]["A"]["cache_hit"])
        self.assertTrue(artifacts_again["creative_features"]["A"]["cache_hit"])
        self.assertIn("analysis", enriched_again["videos"][0])
        self.assertIn("comment_intelligence", enriched["videos"][0]["analysis"])

    def test_enrich_payload_emits_per_video_progress_events(self):
        payload = {
            "videos": [
                {
                    "platform": "youtube",
                    "id": "yt",
                    "public_comment_objects": [{"id": "1", "text": "Nice", "like_count": 2}],
                    "transcript": {"text": "Nice hook."},
                }
            ]
        }
        profiles = build_video_profiles(payload, "demo")
        events = []

        enrich_payload_with_analysis(
            payload,
            profiles=profiles,
            previous_record=None,
            config=AnalysisConfig(comment_intelligence="evidence", creative_features="evidence"),
            progress=lambda stage, message, percent, details: events.append(
                {"stage": stage, "message": message, "percent": percent, "details": details}
            ),
        )

        stages = [event["stage"] for event in events]
        self.assertIn("analysis_comment_intelligence", stages)
        self.assertIn("analysis_comment_intelligence_done", stages)
        self.assertIn("analysis_creative_features", stages)
        self.assertIn("analysis_creative_features_done", stages)
        self.assertTrue(all(event["details"]["video_id"] == "A" for event in events))


if __name__ == "__main__":
    unittest.main()
