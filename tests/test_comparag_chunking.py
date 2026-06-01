import unittest

from comparag.chunking import build_chunks
from comparag.models import sanitize_metadata


class ComparagChunkingTests(unittest.TestCase):
    def test_build_chunks_creates_fact_hook_transcript_and_comment_chunks(self):
        payload = {
            "videos": [
                {
                    "platform": "youtube",
                    "id": "yt",
                    "url": "https://youtube.test/short",
                    "title": "Video A",
                    "creator": "Creator A",
                    "follower_count": 1000,
                    "views": 1000,
                    "likes": 100,
                    "comments": 20,
                    "duration_seconds": 12,
                    "hashtags": ["shorts"],
                    "transcript": {
                        "text": "Hello hook. This is the middle. This is the close.",
                        "segments": [
                            {"start": 0, "duration": 2, "text": "Hello hook."},
                            {"start": 2, "duration": 5, "text": "This is the middle."},
                            {"start": 7, "duration": 5, "text": "This is the close."},
                        ],
                    },
                    "public_comment_objects": [
                        {"text": "Great hook", "like_count": 4, "owner": {"username": "viewer"}}
                    ],
                },
                {
                    "platform": "instagram_post",
                    "id": "ig",
                    "views": 500,
                    "likes": 80,
                    "comments": 10,
                    "duration_seconds": 10,
                    "transcript": {
                        "text": "raw",
                        "variants": {
                            "original": "\u0930\u094b\u091c update",
                            "hinglish_latin": "Roz update",
                            "english_normalized": "Every day update.",
                        },
                        "segments": [{"start": 0, "duration": 10, "text": "\u0930\u094b\u091c update"}],
                    },
                    "public_comment_objects": [],
                    "analysis": {
                        "comment_intelligence": {
                            "available": True,
                            "total_fetched_comments": 2,
                            "useful_comment_count": 1,
                            "cluster_count": 1,
                            "compression_ratio": 0.5,
                            "noise_summary": {"one_word_low_info": 1},
                            "themes": [
                                {
                                    "label": "Repeated CTA replies",
                                    "description": "People repeat a short CTA response.",
                                    "comment_count": 1,
                                    "total_likes": 2,
                                    "examples": ["Info"],
                                }
                            ],
                            "top_clusters": [
                                {
                                    "representative_text": "Info",
                                    "labels": ["one_word_low_info"],
                                    "count": 1,
                                    "total_likes": 2,
                                    "examples": [],
                                }
                            ],
                        },
                        "creative_features": {
                            "available": True,
                            "hook_type": "problem_solution",
                            "first_5s_promise": "Every day update.",
                            "target_audience": "busy operators",
                            "pain_points": ["slow reporting"],
                            "proof_elements": ["workflow demo"],
                            "cta": "try the checklist",
                            "emotional_angle": "relief",
                            "claims": ["faster reporting"],
                            "risk_flags": [],
                            "improvement_opportunities": [],
                        },
                    },
                },
            ]
        }

        profiles, chunks = build_chunks(payload, "demo")
        doc_types = {(chunk.video_id, chunk.doc_type) for chunk in chunks}

        self.assertEqual([profile.video_id for profile in profiles], ["A", "B"])
        self.assertIn(("A", "video_fact_card"), doc_types)
        self.assertIn(("A", "hook_0_5s"), doc_types)
        self.assertIn(("A", "transcript_window"), doc_types)
        self.assertIn(("A", "top_comments"), doc_types)
        self.assertIn(("B", "transcript_text_window"), doc_types)
        self.assertIn(("B", "comment_intelligence_summary"), doc_types)
        self.assertIn(("B", "comment_theme"), doc_types)
        self.assertIn(("B", "comment_cluster"), doc_types)
        self.assertIn(("B", "creative_features"), doc_types)

        b_hook = next(chunk for chunk in chunks if chunk.video_id == "B" and chunk.doc_type == "hook_0_5s")
        self.assertIn("Every", b_hook.text)
        self.assertTrue(b_hook.metadata["is_time_approximate"])

    def test_sanitize_metadata_keeps_chroma_safe_scalars(self):
        metadata = sanitize_metadata(
            {
                "keep": "yes",
                "count": 2,
                "none": None,
                "tags": ["a", "b"],
                "nested": {"skip": True},
            }
        )

        self.assertEqual(metadata, {"keep": "yes", "count": 2, "tags": "a, b"})


if __name__ == "__main__":
    unittest.main()
