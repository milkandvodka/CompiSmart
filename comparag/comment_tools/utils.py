from __future__ import annotations

import re
from typing import Any

from ..models import VideoProfile


def normalize_comment_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w#@]+", " ", text.lower(), flags=re.UNICODE)).strip()


def comment_author(comment: dict[str, Any]) -> str:
    owner = comment.get("owner") if isinstance(comment.get("owner"), dict) else {}
    return str(comment.get("author") or comment.get("username") or owner.get("username") or "unknown").lstrip("@")


def comment_author_url(comment: dict[str, Any], platform: str, author: str) -> str | None:
    url = comment.get("author_url") or comment.get("profile_url")
    if url:
        return str(url)
    if platform.startswith("instagram") and author and author != "unknown":
        return f"https://www.instagram.com/{author}/"
    if platform.startswith("youtube") and author and author != "unknown":
        return f"https://www.youtube.com/{author if author.startswith('@') else '@' + author}"
    return None


def creator_candidates(profile: VideoProfile) -> set[str]:
    return {normalize_username(value) for value in (profile.creator, profile.creator_id) if value}


def normalize_username(value: str | None) -> str:
    return str(value or "").strip().lower().lstrip("@")


def scalar_to_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
