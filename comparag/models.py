from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JsonDict = dict[str, Any]
MetadataValue = str | int | float | bool


@dataclass(frozen=True)
class VideoProfile:
    comparison_id: str
    video_id: str
    platform: str
    source_id: str | None
    url: str | None
    title: str | None
    description: str | None
    creator: str | None
    creator_id: str | None
    creator_url: str | None
    follower_count: int | float | None
    hashtags: list[str]
    upload_date: str | None
    duration_seconds: float | None
    views: int | float | None
    likes: int | float | None
    comments: int | float | None
    fetched_comment_count: int
    engagement_rate: float | None
    thumbnail: str | None

    def to_dict(self) -> JsonDict:
        return {
            "comparison_id": self.comparison_id,
            "video_id": self.video_id,
            "platform": self.platform,
            "source_id": self.source_id,
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "creator": self.creator,
            "creator_id": self.creator_id,
            "creator_url": self.creator_url,
            "follower_count": self.follower_count,
            "hashtags": self.hashtags,
            "upload_date": self.upload_date,
            "duration_seconds": self.duration_seconds,
            "views": self.views,
            "likes": self.likes,
            "comments": self.comments,
            "fetched_comment_count": self.fetched_comment_count,
            "engagement_rate": self.engagement_rate,
            "thumbnail": self.thumbnail,
        }


@dataclass(frozen=True)
class RagChunk:
    id: str
    comparison_id: str
    video_id: str
    doc_type: str
    text: str
    display_text: str
    metadata: dict[str, MetadataValue] = field(default_factory=dict)

    def chroma_metadata(self) -> dict[str, MetadataValue]:
        base: dict[str, MetadataValue] = {
            "comparison_id": self.comparison_id,
            "video_id": self.video_id,
            "doc_type": self.doc_type,
            "chunk_id": self.id,
        }
        base.update(self.metadata)
        return sanitize_metadata(base)

    def citation_label(self) -> str:
        label = self.metadata.get("citation_label")
        if isinstance(label, str) and label:
            return label
        return f"Video {self.video_id}, {self.doc_type}"


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None = None

    @property
    def citation_label(self) -> str:
        label = self.metadata.get("citation_label")
        if isinstance(label, str) and label:
            return label
        video_id = self.metadata.get("video_id", "?")
        doc_type = self.metadata.get("doc_type", "source")
        return f"Video {video_id}, {doc_type}"


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, MetadataValue]:
    """Keep Chroma metadata scalar-only and drop empty values."""

    sanitized: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, bool):
            sanitized[key] = value
        elif isinstance(value, (str, int, float)):
            sanitized[key] = value
        elif isinstance(value, (list, tuple, set)):
            joined = ", ".join(str(item) for item in value if item not in (None, ""))
            if joined:
                sanitized[key] = joined
        elif isinstance(value, dict):
            continue
        else:
            sanitized[key] = str(value)
    return sanitized
