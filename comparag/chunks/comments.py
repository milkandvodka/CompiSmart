from __future__ import annotations

from ..models import JsonDict, RagChunk, VideoProfile
from .utils import chunk_id, grouped

COMMENT_GROUP_SIZE = 10


def build_comment_chunks(video: JsonDict, profile: VideoProfile) -> list[RagChunk]:
    comments = video.get("public_comment_objects") or []
    normalized_comments = [comment for comment in comments if isinstance(comment, dict) and comment.get("text")]
    if not normalized_comments:
        return []

    normalized_comments.sort(key=lambda item: int(item.get("like_count") or item.get("likes_count") or 0), reverse=True)
    groups = list(grouped(normalized_comments[:50], COMMENT_GROUP_SIZE))
    chunks: list[RagChunk] = []
    for index, group in enumerate(groups):
        lines = []
        for comment in group:
            username = comment_username(comment)
            likes = comment.get("like_count", comment.get("likes_count"))
            like_text = f"{likes} likes" if likes is not None else "unknown likes"
            lines.append(f"- @{username}: {comment.get('text')} ({like_text})")
        text = f"Video {profile.video_id} top public comments, group {index + 1}.\n" + "\n".join(lines)
        chunks.append(
            RagChunk(
                id=chunk_id(profile, "top_comments", str(index)),
                comparison_id=profile.comparison_id,
                video_id=profile.video_id,
                doc_type="top_comments",
                text=text,
                display_text=text,
                metadata={
                    "platform": profile.platform,
                    "source_url": profile.url,
                    "comment_group_index": index,
                    "comment_count": len(group),
                    "citation_label": f"Video {profile.video_id}, top comments group {index + 1}",
                },
            )
        )
    return chunks


def comment_username(comment: JsonDict) -> str:
    owner = comment.get("owner") or comment.get("author") or {}
    if isinstance(owner, dict):
        username = owner.get("username") or owner.get("id")
        if username:
            return str(username)
    return "unknown"
