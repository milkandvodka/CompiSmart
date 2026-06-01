from __future__ import annotations

from typing import Any

from .models import JsonDict, VideoProfile


VIDEO_IDS = ("A", "B")


def number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        try:
            parsed = float(cleaned)
        except ValueError:
            return None
        return int(parsed) if parsed.is_integer() else parsed
    return None


def compute_engagement_rate(
    *,
    likes: int | float | None,
    comments: int | float | None,
    views: int | float | None,
) -> float | None:
    if views in (None, 0) or likes is None or comments is None:
        return None
    return round(((likes + comments) / views) * 100, 4)


def public_comment_count(video: JsonDict) -> int | float | None:
    comments = number_or_none(video.get("comments"))
    if comments is not None:
        return comments
    return number_or_none(video.get("public_comment_object_count"))


def build_video_profiles(payload: JsonDict, comparison_id: str) -> list[VideoProfile]:
    videos = payload.get("videos") or []
    profiles: list[VideoProfile] = []

    for index, video in enumerate(videos[:2]):
        video_id = VIDEO_IDS[index] if index < len(VIDEO_IDS) else str(index + 1)
        views = number_or_none(video.get("views"))
        likes = number_or_none(video.get("likes"))
        comments = public_comment_count(video)
        engagement_rate = compute_engagement_rate(likes=likes, comments=comments, views=views)
        hashtags = video.get("hashtags") or []
        if not isinstance(hashtags, list):
            hashtags = []

        profiles.append(
            VideoProfile(
                comparison_id=comparison_id,
                video_id=video_id,
                platform=str(video.get("platform") or ""),
                source_id=maybe_str(video.get("id")),
                url=maybe_str(video.get("url")),
                title=maybe_str(video.get("title")),
                description=maybe_str(video.get("description")),
                creator=maybe_str(video.get("creator")),
                creator_id=maybe_str(video.get("creator_id")),
                creator_url=maybe_str(video.get("creator_url")),
                follower_count=number_or_none(video.get("follower_count")),
                hashtags=[str(tag) for tag in hashtags],
                upload_date=maybe_str(video.get("upload_date")),
                duration_seconds=number_or_none(video.get("duration_seconds")),
                views=views,
                likes=likes,
                comments=comments,
                fetched_comment_count=int(number_or_none(video.get("public_comment_object_count")) or 0),
                engagement_rate=engagement_rate,
                thumbnail=maybe_str(video.get("thumbnail")),
            )
        )

    return profiles


def maybe_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def format_metric(value: int | float | None, *, suffix: str = "") -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.2f}{suffix}"
    return f"{int(value):,}{suffix}"
