import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from social_video_extractor import (
    URLValidationError,
    TranscriptResult,
    detect_platform,
    extract_instagram_shortcode,
    extract_youtube_video_id,
    instagram_fallback_info_from_supplement,
    instagram_supplement_has_media,
    instagram_sessionid_from_cookie_file,
    is_instagram_empty_media_error,
    normalize_thumbnails,
    parse_json3_caption,
    parse_vtt_or_srt_caption,
    resolve_transcript,
    resolve_input_urls,
    collect_hashtags,
    normalize_comments,
    normalize_instagrapi_comment,
    comment_source_summary,
    normalize_instagram_raw_comment,
    fetch_fast_instagram_comments,
    choose_asr_provider,
    extract_transcript,
    asr_transcript_is_suspicious,
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

    @patch("social_video_extractor.fetch_caption_text")
    def test_extract_transcript_prefers_raw_variant_before_translated_caption(self, fetch_mock):
        translated_url = "https://www.youtube.com/api/timedtext?lang=hi&tlang=en&fmt=json3"
        raw_url = "https://www.youtube.com/api/timedtext?lang=hi&fmt=json3"

        def fake_fetch(url):
            self.assertEqual(url, raw_url)
            return '{"events":[{"tStartMs":0,"dDurationMs":2000,"segs":[{"utf8":"raw caption text"}]}]}'

        fetch_mock.side_effect = fake_fetch

        result = extract_transcript(
            {
                "automatic_captions": {
                    "en": [{"url": translated_url, "ext": "json3"}],
                }
            },
            "en",
        )

        self.assertTrue(result.available)
        self.assertEqual(result.text, "raw caption text")
        self.assertEqual(result.source, raw_url)
        self.assertEqual(fetch_mock.call_count, 1)

    @patch("social_video_extractor.fetch_caption_text")
    def test_extract_transcript_falls_back_to_translated_when_raw_caption_fails(self, fetch_mock):
        translated_url = "https://www.youtube.com/api/timedtext?lang=hi&tlang=en&fmt=json3"
        raw_url = "https://www.youtube.com/api/timedtext?lang=hi&fmt=json3"

        def fake_fetch(url):
            if url == raw_url:
                raise RuntimeError("raw caption failed")
            self.assertEqual(url, translated_url)
            return '{"events":[{"tStartMs":0,"dDurationMs":2000,"segs":[{"utf8":"translated caption text"}]}]}'

        fetch_mock.side_effect = fake_fetch

        result = extract_transcript(
            {
                "automatic_captions": {
                    "en": [{"url": translated_url, "ext": "json3"}],
                }
            },
            "en",
        )

        self.assertTrue(result.available)
        self.assertEqual(result.text, "translated caption text")
        self.assertEqual(result.source, translated_url)

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

    def test_short_asr_for_long_video_is_suspicious(self):
        result = TranscriptResult(True, "tiny output", [], None, "audio.wav", "asr")

        self.assertTrue(asr_transcript_is_suspicious(result, {"duration": 60}))

    @patch("social_video_extractor.transcribe_audio")
    @patch("social_video_extractor.download_audio_for_asr")
    def test_resolve_transcript_retries_local_when_hosted_asr_is_suspicious(self, download_mock, transcribe_mock):
        download_mock.return_value = Path("audio.wav")
        transcribe_mock.side_effect = [
            TranscriptResult(True, "tiny output", [], None, "audio.wav", "asr", engine="huggingface-inference"),
            TranscriptResult(
                True,
                "A long enough local transcript with enough words to pass the duration-based quality gate safely.",
                [],
                "hi",
                "audio.wav",
                "asr",
                engine="faster-whisper",
            ),
        ]

        result = resolve_transcript(
            info={"id": "yt", "duration": 60},
            url="https://www.youtube.com/watch?v=yt",
            platform="youtube",
            preferred_caption_language="en",
            transcribe_missing=True,
            asr_provider="auto",
            asr_model="base",
            hf_asr_model="openai/whisper-large-v3-turbo",
            hf_token="token",
            asr_timeout_seconds=60,
            asr_language=None,
            asr_device="cpu",
            asr_compute_type="int8",
            asr_cache_dir=Path(".cache"),
            keep_audio=True,
            cookies=None,
            cookies_from_browser=None,
        )

        self.assertTrue(result.available)
        self.assertEqual(result.engine, "faster-whisper")
        self.assertEqual(transcribe_mock.call_count, 2)
        self.assertEqual(transcribe_mock.call_args_list[1].kwargs["provider"], "local")

    @patch("social_video_extractor.transcribe_audio")
    @patch("social_video_extractor.download_audio_for_asr")
    def test_resolve_transcript_drops_suspicious_asr_when_local_retry_is_suspicious(self, download_mock, transcribe_mock):
        download_mock.return_value = Path("audio.wav")
        transcribe_mock.side_effect = [
            TranscriptResult(True, "tiny output", [], None, "audio.wav", "asr", engine="huggingface-inference"),
            TranscriptResult(True, "short again", [], "hi", "audio.wav", "asr", engine="faster-whisper"),
        ]

        result = resolve_transcript(
            info={"id": "yt", "duration": 60},
            url="https://www.youtube.com/watch?v=yt",
            platform="youtube",
            preferred_caption_language="en",
            transcribe_missing=True,
            asr_provider="auto",
            asr_model="base",
            hf_asr_model="openai/whisper-large-v3-turbo",
            hf_token="token",
            asr_timeout_seconds=60,
            asr_language=None,
            asr_device="cpu",
            asr_compute_type="int8",
            asr_cache_dir=Path(".cache"),
            keep_audio=True,
            cookies=None,
            cookies_from_browser=None,
        )

        self.assertFalse(result.available)
        self.assertEqual(result.text, "")
        self.assertIn("not indexed as reliable", result.note)

    def test_instagram_sessionid_from_cookie_file_reads_netscape_cookie(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cookies.txt"
            path.write_text(
                "# Netscape HTTP Cookie File\n"
                "#HttpOnly_.instagram.com\tTRUE\t/\tTRUE\t1893456000\tsessionid\tabc123\n",
                encoding="utf-8",
            )

            self.assertEqual(instagram_sessionid_from_cookie_file(path), "abc123")

    def test_detects_instagram_empty_media_error(self):
        self.assertTrue(is_instagram_empty_media_error(Exception("[Instagram] ABC: Instagram sent an empty media response")))
        self.assertFalse(is_instagram_empty_media_error(Exception("[YouTube] unavailable")))

    def test_instagram_fallback_info_from_instagrapi_supplement(self):
        supplement = {
            "available": False,
            "instagrapi": {
                "available": True,
                "media_id": "12345",
                "media_info": {
                    "caption_text": "Caption #tag",
                    "video_url": "https://cdninstagram.com/video.mp4",
                    "thumbnail_url": "https://cdninstagram.com/thumb.jpg",
                    "video_duration": 12.5,
                    "play_count": 1000,
                    "like_count": 100,
                    "comment_count": 10,
                    "taken_at": "2026-06-01T10:00:00+00:00",
                    "user": {"username": "creator", "pk": "9"},
                },
                "user_info": {"username": "creator", "full_name": "Creator", "follower_count": 2000},
            },
        }

        info = instagram_fallback_info_from_supplement("https://www.instagram.com/reel/ABC123/", supplement)

        self.assertTrue(instagram_supplement_has_media(supplement))
        self.assertEqual(info["id"], "12345")
        self.assertEqual(info["description"], "Caption #tag")
        self.assertEqual(info["direct_media_url"], "https://cdninstagram.com/video.mp4")
        self.assertEqual(info["thumbnail"], "https://cdninstagram.com/thumb.jpg")
        self.assertEqual(info["duration"], 12.5)
        self.assertEqual(info["view_count"], 1000)
        self.assertEqual(info["uploader_url"], "https://www.instagram.com/creator/")
        self.assertEqual(info["formats"][0]["format_id"], "instagrapi_video")

    @patch("social_video_extractor.transcribe_audio")
    @patch("social_video_extractor.download_audio_for_asr")
    def test_resolve_transcript_uses_direct_media_url_for_asr(self, download_mock, transcribe_mock):
        download_mock.return_value = Path("audio.mp4")
        transcribe_mock.return_value = TranscriptResult(True, "hello", [], "en", "audio.mp4", "asr")

        result = resolve_transcript(
            info={"id": "ig", "direct_media_url": "https://cdninstagram.com/video.mp4"},
            url="https://www.instagram.com/reel/ABC123/",
            platform="instagram_reel",
            preferred_caption_language="en",
            transcribe_missing=True,
            asr_provider="local",
            asr_model="base",
            hf_asr_model="openai/whisper-large-v3-turbo",
            hf_token=None,
            asr_timeout_seconds=60,
            asr_language=None,
            asr_device="cpu",
            asr_compute_type="int8",
            asr_cache_dir=Path(".cache"),
            keep_audio=True,
            cookies=None,
            cookies_from_browser=None,
        )

        self.assertTrue(result.available)
        self.assertEqual(download_mock.call_args.args[0], "https://cdninstagram.com/video.mp4")


if __name__ == "__main__":
    unittest.main()
