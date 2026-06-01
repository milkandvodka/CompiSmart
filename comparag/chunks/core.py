from __future__ import annotations

from ..metrics import build_video_profiles, format_metric
from ..models import JsonDict, RagChunk, VideoProfile
from .analysis import build_analysis_chunks
from .comments import build_comment_chunks
from .transcripts import build_transcript_chunks
from .utils import chunk_id


def build_chunks(payload: JsonDict, comparison_id: str) -> tuple[list[VideoProfile], list[RagChunk]]:
    profiles = build_video_profiles(payload, comparison_id)
    chunks: list[RagChunk] = []
    videos = payload.get("videos") or []

    for profile, video in zip(profiles, videos):
        chunks.append(build_fact_card_chunk(profile))
        chunks.extend(build_transcript_chunks(video, profile))
        chunks.extend(build_analysis_chunks(video, profile))
        chunks.extend(build_comment_chunks(video, profile))

    return profiles, chunks


def build_fact_card_chunk(profile: VideoProfile) -> RagChunk:
    engagement = "unknown" if profile.engagement_rate is None else f"{profile.engagement_rate:.2f}%"
    hashtags = ", ".join(profile.hashtags[:20]) if profile.hashtags else "none"
    text = "\n".join(
        [
            f"Video {profile.video_id} metadata snapshot.",
            f"Platform: {profile.platform}",
            f"Title: {profile.title or 'unknown'}",
            f"Creator: {profile.creator or 'unknown'}",
            f"Creator follower count: {format_metric(profile.follower_count)}",
            f"Views: {format_metric(profile.views)}",
            f"Likes: {format_metric(profile.likes)}",
            f"Comments: {format_metric(profile.comments)}",
            f"Engagement rate: {engagement}",
            f"Upload date: {profile.upload_date or 'unknown'}",
            f"Duration seconds: {format_metric(profile.duration_seconds)}",
            f"Hashtags: {hashtags}",
        ]
    )
    return RagChunk(
        id=chunk_id(profile, "fact_card", "metadata"),
        comparison_id=profile.comparison_id,
        video_id=profile.video_id,
        doc_type="video_fact_card",
        text=text,
        display_text=text,
        metadata={
            "platform": profile.platform,
            "source_url": profile.url,
            "creator": profile.creator,
            "views": profile.views,
            "likes": profile.likes,
            "comments": profile.comments,
            "engagement_rate": profile.engagement_rate,
            "citation_label": f"Video {profile.video_id}, metadata snapshot",
        },
    )
