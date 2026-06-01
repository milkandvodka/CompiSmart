import unittest

from comparag.metrics import build_video_profiles, compute_engagement_rate


class ComparagMetricsTests(unittest.TestCase):
    def test_compute_engagement_rate_uses_likes_comments_over_views(self):
        self.assertEqual(compute_engagement_rate(likes=100, comments=25, views=1000), 12.5)

    def test_compute_engagement_rate_returns_none_without_views(self):
        self.assertIsNone(compute_engagement_rate(likes=100, comments=25, views=0))

    def test_build_video_profiles_assigns_a_b_and_prefers_public_comment_count(self):
        payload = {
            "videos": [
                {
                    "platform": "youtube",
                    "id": "yt",
                    "views": 1000,
                    "likes": 100,
                    "comments": 50,
                    "public_comment_object_count": 10,
                },
                {
                    "platform": "instagram_post",
                    "id": "ig",
                    "views": 200,
                    "likes": 20,
                    "public_comment_object_count": 5,
                },
            ]
        }

        profiles = build_video_profiles(payload, "demo")

        self.assertEqual([profile.video_id for profile in profiles], ["A", "B"])
        self.assertEqual(profiles[0].comments, 50)
        self.assertEqual(profiles[0].engagement_rate, 15.0)
        self.assertEqual(profiles[1].comments, 5)
        self.assertEqual(profiles[1].engagement_rate, 12.5)


if __name__ == "__main__":
    unittest.main()
