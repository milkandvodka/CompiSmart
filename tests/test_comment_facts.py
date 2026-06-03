import unittest

from comparag.comment_facts import build_comment_fact_table, query_comment_facts
from comparag.metrics import build_video_profiles


class CommentFactsTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "videos": [
                {
                    "platform": "youtube",
                    "id": "yt",
                    "creator": "YT Creator",
                    "public_comment_objects": [
                        {
                            "id": "yt1",
                            "text": "Alpha note",
                            "like_count": 7,
                            "author": "@viewer",
                            "author_id": "yt-user-1",
                            "author_url": "https://www.youtube.com/@viewer",
                        },
                        {
                            "id": "yt2",
                            "text": "alpha note!",
                            "like_count": 3,
                            "author": "@viewer2",
                            "author_id": "yt-user-2",
                        },
                    ],
                },
                {
                    "platform": "instagram_post",
                    "id": "ig",
                    "creator": "brand",
                    "public_comment_objects": [
                        {"id": "ig1", "text": "Beta code", "author": "buyer", "author_id": "ig-user-1"},
                        {"id": "ig2", "text": "Beta code please", "like_count": 2, "author": "buyer2", "author_id": "ig-user-2"},
                    ],
                },
            ]
        }
        self.profiles = build_video_profiles(self.payload, "demo")
        self.table = build_comment_fact_table(self.payload, self.profiles)

    def test_build_comment_fact_table_preserves_author_ids_and_urls(self):
        facts = self.table["facts"]

        self.assertEqual(len(facts), 4)
        self.assertEqual(facts[0]["author_id"], "yt-user-1")
        self.assertEqual(facts[0]["author_url"], "https://www.youtube.com/@viewer")
        self.assertEqual(facts[2]["author_url"], "https://www.instagram.com/buyer/")

    def test_query_phrase_matches_counts_and_sums_likes(self):
        result = query_comment_facts('Who commented "beta code" in the insta video?', self.table)

        self.assertTrue(result["available"])
        self.assertIn("2 matching comments", result["answer_text"])
        self.assertIn("2 total comment likes", result["answer_text"])
        self.assertIn("ig-user-1", result["answer_text"])

    def test_query_phrase_matches_multiple_phrases_maps_insta_then_youtube(self):
        result = query_comment_facts('Who commented "beta code" in the insta video and "alpha note" in the yt video?', self.table)

        self.assertTrue(result["available"])
        self.assertIn("Phrase 'beta code' in Video B", result["answer_text"])
        self.assertIn("Phrase 'alpha note' in Video A", result["answer_text"])
        self.assertIn("10 total comment likes", result["answer_text"])

    def test_query_most_liked_commenters(self):
        result = query_comment_facts("Which user profile had the most likes in comments?", self.table)

        self.assertTrue(result["available"])
        self.assertIn("@viewer", result["answer_text"])
        self.assertIn("yt-user-1", result["answer_text"])

    def test_query_most_comment_likes_phrase(self):
        result = query_comment_facts("Give me the profile of the fetched commenter with the most comment likes", self.table)

        self.assertTrue(result["available"])
        self.assertEqual(result["tool"], "comment_top_commenters")
        self.assertIn("@viewer", result["answer_text"])

    def test_query_top_instagram_comments_phrase(self):
        result = query_comment_facts("Show me the top Instagram comments with profile URLs", self.table)

        self.assertTrue(result["available"])
        self.assertEqual(result["tool"], "comment_top_comments")
        self.assertIn("https://www.instagram.com/buyer2/", result["answer_text"])

    def test_query_who_wrote_top_liked_comment_uses_ranked_comment_facts(self):
        result = query_comment_facts("Who wrote the top liked comment and what was the like count?", self.table)

        self.assertTrue(result["available"])
        self.assertEqual(result["tool"], "comment_top_comments")
        self.assertIn("@viewer", result["answer_text"])
        self.assertIn("7 likes", result["answer_text"])

    def test_query_every_fetched_instagram_comment_uses_comment_facts(self):
        result = query_comment_facts(
            "For Instagram Video B only, list every fetched comment with username, profile URL, comment-like count, and text.",
            self.table,
        )

        self.assertTrue(result["available"])
        self.assertEqual(result["tool"], "comment_top_comments")
        self.assertIn("buyer", result["answer_text"])
        self.assertIn("https://www.instagram.com/buyer/", result["answer_text"])

    def test_query_unquoted_keyword_list_searches_all_videos(self):
        result = query_comment_facts(
            "Who commented about alpha, beta, or missing? Give exact count and total likes.",
            self.table,
        )

        self.assertTrue(result["available"])
        self.assertIn("Phrase 'alpha': 2 matching comments", result["answer_text"])
        self.assertIn("10 total comment likes", result["answer_text"])
        self.assertIn("Phrase 'beta': 2 matching comments", result["answer_text"])
        self.assertIn("2 total comment likes", result["answer_text"])

    def test_query_single_most_liked_user_returns_one_row(self):
        result = query_comment_facts("Give me the profile of the user who had the most likes in comments", self.table)

        self.assertTrue(result["available"])
        self.assertEqual(len(result["facts"]), 1)

    def test_follow_up_phrase_resolves_from_history(self):
        result = query_comment_facts(
            "What were their total comment likes again?",
            self.table,
            history=[{"role": "assistant", "content": "Phrase 'beta code': 2 matching comments, 2 total comment likes."}],
        )

        self.assertTrue(result["available"])
        self.assertIn("Phrase 'beta code'", result["answer_text"])
        self.assertIn("2 total comment likes", result["answer_text"])

    def test_follow_up_phrase_resolves_from_curly_quoted_answer(self):
        result = query_comment_facts(
            "What were their total comment likes again?",
            self.table,
            history=[{"role": "assistant", "content": "In Video B, 2 commenters wrote “Beta code,” with 2 total likes."}],
        )

        self.assertTrue(result["available"])
        self.assertIn("Phrase 'Beta code'", result["answer_text"])
        self.assertIn("2 total comment likes", result["answer_text"])

    def test_follow_up_phrase_resolves_from_matching_answer(self):
        result = query_comment_facts(
            "What were their total comment likes again?",
            self.table,
            history=[{"role": "assistant", "content": "Found 2 Instagram comments matching “beta code,” with 2 total likes."}],
        )

        self.assertTrue(result["available"])
        self.assertIn("Phrase 'beta code'", result["answer_text"])


    def test_follow_up_resolves_exact_comment_ids_from_history(self):
        result = query_comment_facts(
            "For those same commenters, give me their profile URLs and user IDs only.",
            self.table,
            history=[
                {
                    "role": "assistant",
                    "content": "Matched @buyer [Video B, comment ig1] and @buyer2 [Video B, comment ig2].",
                }
            ],
        )

        self.assertTrue(result["available"])
        self.assertEqual(result["tool"], "comment_id_followup")
        self.assertIn("ig-user-1", result["answer_text"])
        self.assertIn("https://www.instagram.com/buyer/", result["answer_text"])


if __name__ == "__main__":
    unittest.main()
