import argparse
import unittest

from social_video_extractor import (
    URLValidationError,
    detect_platform,
    extract_instagram_shortcode,
    extract_youtube_video_id,
    normalize_thumbnails,
    parse_json3_caption,
    parse_vtt_or_srt_caption,
    resolve_input_urls,
    collect_hashtags,
    normalize_comments,
    normalize_instagrapi_comment,
    comment_source_summary,
    normalize_instagram_raw_comment,
    fetch_fast_instagram_comments,
    choose_asr_provider,
)


class SocialVideoExtractorTests(unittest.TestCase):
    def test_detect_platform_accepts_required_platforms(self):
        self.assertEqual(detect_platform("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "youtube")
        self.assertEqual(detect_platform("https://youtu.be/dQw4w9WgXcQ"), "youtube")
        self.assertEqual(detect_platform("https://www.instagram.com/reel/ABC123/"), "instagram_reel")
        self.assertEqual(detect_platform("https://www.instagram.com/p/ABC123/"), "instagram_post")

    def test_detect_platform_rejects_instagram_profile(self):
        self.assertIsNone(detect_platform("https://www.instagram.com/openai/"))

    def test_extract_youtube_video_id_supports_common_shapes(self):
        self.assertEqual(
            extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )
        self.assertEqual(extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(
            extract_youtube_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )

    def test_extract_instagram_shortcode_supports_posts_and_reels(self):
        self.assertEqual(extract_instagram_shortcode("https://www.instagram.com/p/DXtcOwnDC5k/"), "DXtcOwnDC5k")
        self.assertEqual(extract_instagram_shortcode("https://www.instagram.com/reel/ABC123/"), "ABC123")

    def test_resolve_input_urls_allows_positional_urls_in_either_order(self):
        args = argparse.Namespace(
            youtube_url=None,
            instagram_url=None,
            urls=[
                "https://www.instagram.com/p/ABC123/",
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            ],
        )

        youtube_url, instagram_url = resolve_input_urls(args)

        self.assertEqual(youtube_url, "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertEqual(instagram_url, "https://www.instagram.com/p/ABC123/")

    def test_resolve_input_urls_requires_both_named_inputs(self):
        args = argparse.Namespace(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            instagram_url=None,
            urls=[],
        )

        with self.assertRaises(URLValidationError):
            resolve_input_urls(args)

    def test_parse_json3_caption(self):
        segments = parse_json3_caption(
            """
            {
              "events": [
                {"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "Hello"}, {"utf8": " world"}]},
                {"tStartMs": 3500, "dDurationMs": 1000, "segs": [{"utf8": "\\n"}]}
              ]
            }
            """
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start, 1.0)
        self.assertEqual(segments[0].duration, 2.0)
        self.assertEqual(segments[0].text, "Hello world")

    def test_parse_vtt_caption(self):
        segments = parse_vtt_or_srt_caption(
            """WEBVTT

00:00:01.000 --> 00:00:03.000
<c>Hello</c> world

00:00:03.500 --> 00:00:04.000
Again
"""
        )

        self.assertEqual([segment.text for segment in segments], ["Hello world", "Again"])
        self.assertEqual(segments[0].start, 1.0)
        self.assertEqual(segments[0].duration, 2.0)

    def test_collect_hashtags_reads_title_description_and_tags(self):
        hashtags = collect_hashtags(
            {
                "title": "Clip #Shorts",
                "description": "More #funny",
                "tags": ["#Trending", "creator"],
            }
        )

        self.assertEqual(hashtags, ["creator", "funny", "Shorts", "Trending"])

    def test_normalize_thumbnails_keeps_urls(self):
        thumbnails = normalize_thumbnails(
            {
                "thumbnail": "https://example.test/main.jpg",
                "thumbnails": [{"url": "https://example.test/thumb.jpg", "width": 120, "height": 90}],
            },
            {"available": True, "display_url": "https://example.test/ig.jpg"},
        )

        self.assertEqual(
            [thumbnail["url"] for thumbnail in thumbnails],
            ["https://example.test/thumb.jpg", "https://example.test/main.jpg", "https://example.test/ig.jpg"],
        )

    def test_normalize_thumbnails_reads_instagrapi_candidates(self):
        thumbnails = normalize_thumbnails(
            {},
            {
                "instagrapi": {
                    "media_info": {
                        "thumbnail_url": "https://example.test/main.jpg",
                        "image_versions2": {
                            "candidates": [
                                {"url": "https://example.test/a.jpg", "width": 1080, "height": 1080}
                            ]
                        },
                    }
                }
            },
        )

        self.assertEqual(
            [thumbnail["url"] for thumbnail in thumbnails],
            ["https://example.test/main.jpg", "https://example.test/a.jpg"],
        )

    def test_normalize_instagrapi_comment_includes_like_count(self):
        comment = normalize_instagrapi_comment(
            {
                "pk": "123",
                "text": "Nice",
                "like_count": 7,
                "user": {"username": "viewer"},
            }
        )

        self.assertEqual(comment["source"], "instagrapi")
        self.assertEqual(comment["id"], "123")
        self.assertEqual(comment["like_count"], 7)
        self.assertEqual(comment["owner"]["username"], "viewer")

    def test_normalize_comments_merges_sources_and_dedupes(self):
        comments = normalize_comments(
            {"comments": [{"id": "1", "text": "yt"}]},
            {
                "available": True,
                "comment_objects": [{"id": "2", "text": "loader"}],
                "instagrapi": {
                    "available": True,
                    "comment_objects": [
                        {"id": "2", "text": "dupe"},
                        {"id": "3", "text": "private"},
                        {"id": "4", "text": "reply", "parent_comment_id": "3", "source": "instagrapi_reply"},
                    ],
                },
            },
        )

        self.assertEqual([comment["id"] for comment in comments], ["1", "2", "3", "4"])
        self.assertEqual(
            [comment["source"] for comment in comments],
            ["yt-dlp", "instaloader", "instagrapi", "instagrapi_reply"],
        )
        self.assertEqual(comments[-1]["parent_comment_id"], "3")

    def test_normalize_comments_merges_duplicate_with_like_count(self):
        comments = normalize_comments(
            {"comments": [{"id": "1", "text": "yt"}]},
            {
                "available": True,
                "instagrapi": {
                    "available": True,
                    "comment_objects": [{"id": "1", "text": "yt", "like_count": 0}],
                },
            },
        )

        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0]["source"], "instagrapi")
        self.assertEqual(comments[0]["like_count"], 0)
        self.assertEqual(comments[0]["sources"], ["yt-dlp", "instagrapi"])

    def test_comment_source_summary_reports_instagrapi_error(self):
        summary = comment_source_summary(
            {"comments": []},
            {"available": False, "error": "blocked", "instagrapi": {"available": False, "error": "login required"}},
        )

        self.assertEqual(summary["instaloader"]["error"], "blocked")
        self.assertEqual(summary["instagrapi"]["error"], "login required")

    def test_normalize_instagram_raw_comment_maps_comment_like_count(self):
        comment = normalize_instagram_raw_comment(
            {
                "pk": "10",
                "text": "reply",
                "comment_like_count": 4,
                "created_at_utc": 1778206916,
                "user": {"username": "brand"},
            }
        )

        self.assertEqual(comment["id"], "10")
        self.assertEqual(comment["like_count"], 4)
        self.assertEqual(comment["likes_count"], 4)
        self.assertEqual(comment["owner"]["username"], "brand")
        self.assertTrue(comment["created_at_utc"].startswith("2026-"))

    def test_normalize_instagram_raw_comment_keeps_zero_like_count(self):
        comment = normalize_instagram_raw_comment({"pk": "10", "comment_like_count": 0})

        self.assertEqual(comment["like_count"], 0)

    def test_fetch_fast_instagram_comments_includes_preview_replies(self):
        class FakeClient:
            def private_request(self, endpoint, params=None):
                self.endpoint = endpoint
                self.params = params
                return {
                    "comments": [
                        {
                            "pk": "1",
                            "text": "parent",
                            "comment_like_count": 2,
                            "user": {"username": "parent_user"},
                            "preview_child_comments": [
                                {
                                    "pk": "2",
                                    "text": "reply",
                                    "comment_like_count": 1,
                                    "user": {"username": "reply_user"},
                                }
                            ],
                        }
                    ],
                    "has_more_headload_comments": False,
                }

        comments, summary = fetch_fast_instagram_comments(
            FakeClient(),
            "media_id",
            max_comments=10,
            time_budget_seconds=None,
        )

        self.assertEqual(summary["top_level_count"], 1)
        self.assertEqual(summary["preview_reply_count"], 1)
        self.assertEqual([comment["source"] for comment in comments], ["instagrapi", "instagrapi_preview_reply"])
        self.assertEqual(comments[1]["parent_comment_id"], "1")

    def test_choose_asr_provider_prefers_hf_when_token_available(self):
        self.assertEqual(choose_asr_provider("auto", "token"), "hf")
        self.assertEqual(choose_asr_provider("auto", None), "local")
        self.assertEqual(choose_asr_provider("local", "token"), "local")


if __name__ == "__main__":
    unittest.main()
