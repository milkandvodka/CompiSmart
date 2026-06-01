import unittest
from unittest.mock import patch

from comparag.comment_intelligence import (
    analyze_comments,
    build_comment_clusters,
    build_comment_evidence_pack,
    normalize_comment,
)


class CommentIntelligenceTests(unittest.TestCase):
    def test_normalize_comment_preserves_mechanical_labels_and_creator_reply(self):
        comment = {"id": "1", "text": "Alpha code", "like_count": 3, "owner": {"username": "brand"}}

        normalized = normalize_comment(comment, creator_username="brand")

        self.assertIn("creator_reply", normalized["labels"])
        self.assertNotIn("semantic_positive", normalized["labels"])
        self.assertEqual(normalized["like_count"], 3)

    def test_clusters_merge_same_words_with_symbol_variants_and_use_likes(self):
        comments = [
            normalize_comment({"id": "1", "text": "Update", "like_count": 1}, creator_username=None),
            normalize_comment({"id": "2", "text": "Update !!!", "like_count": 4}, creator_username=None),
        ]

        clusters = build_comment_clusters(comments)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["count"], 2)
        self.assertEqual(clusters[0]["total_likes"], 5)
        self.assertEqual(clusters[0]["representative_text"], "Update !!!")

    def test_evidence_pack_respects_budget(self):
        clusters = [
            {
                "labels": ["unclassified"],
                "count": 10,
                "total_likes": 5,
                "representative_text": "x" * 500,
            }
            for _ in range(20)
        ]

        pack = build_comment_evidence_pack(
            video_id="A",
            total_comments=20,
            clusters=clusters,
            noise={},
            max_chars=700,
        )

        self.assertLessEqual(len(pack), 700)
        self.assertIn("Video A compressed comment evidence", pack)

    @patch("comparag.comment_intelligence.gemini_json")
    def test_llm_comment_intelligence_uses_compressed_pack(self, gemini_mock):
        gemini_mock.return_value = {
            "audience_takeaways": ["People repeat the same short request."],
            "themes": [{"label": "Repeated short requests", "description": "Many short repeated requests."}],
            "objections": [],
            "recommended_creator_actions": ["Make the next step clearer."],
            "notes": [],
        }

        result = analyze_comments(
            [{"id": "1", "text": "Alpha code please", "like_count": 4}],
            video_id="B",
            creator_username=None,
            mode="llm",
            model="test-model",
        )

        self.assertTrue(result["llm_used"])
        self.assertEqual(result["themes"][0]["label"], "Repeated short requests")
        self.assertIn("compressed comment evidence", gemini_mock.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
