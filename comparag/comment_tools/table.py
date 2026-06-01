from __future__ import annotations

from typing import Any

from ..models import JsonDict, VideoProfile
from .constants import COMMENT_FACT_SCHEMA_VERSION
from .utils import (
    as_int,
    comment_author,
    comment_author_url,
    creator_candidates,
    normalize_comment_text,
    normalize_username,
    scalar_to_str,
)


def build_comment_fact_table(payload: JsonDict, profiles: list[VideoProfile]) -> dict[str, Any]:
    videos = payload.get("videos") or []
    facts: list[dict[str, Any]] = []
    for profile, video in zip(profiles, videos):
        creator_usernames = creator_candidates(profile)
        for index, comment in enumerate(video.get("public_comment_objects") or []):
            if not isinstance(comment, dict) or not (comment.get("text") or "").strip():
                continue
            text = str(comment.get("text") or "").strip()
            author = comment_author(comment)
            fact = {
                "video_id": profile.video_id,
                "platform": profile.platform,
                "comment_id": str(comment.get("id") or f"{profile.video_id}-{index}"),
                "text": text,
                "normalized_text": normalize_comment_text(text),
                "author_username": author,
                "author_id": scalar_to_str(comment.get("author_id") or comment.get("owner_id") or comment.get("user_id")),
                "author_url": comment_author_url(comment, profile.platform, author),
                "like_count": as_int(comment.get("like_count", comment.get("likes_count"))),
                "reply_count": as_int(comment.get("reply_count")),
                "is_creator_reply": normalize_username(author) in creator_usernames,
                "is_pinned": bool(comment.get("is_pinned")),
                "timestamp": comment.get("timestamp"),
                "source": comment.get("source"),
            }
            facts.append(fact)
    return {"schema_version": COMMENT_FACT_SCHEMA_VERSION, "facts": facts}
