from __future__ import annotations

from typing import Any

from ..metrics import format_metric


def comment_citation(fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": comment_citation_label(fact),
        "video_id": fact.get("video_id"),
        "doc_type": "comment_fact",
        "comment_id": fact.get("comment_id"),
    }


def comment_citation_label(fact: dict[str, Any]) -> str:
    return f"Video {fact.get('video_id')}, comment {fact.get('comment_id')}"


def format_comment_fact_line(fact: dict[str, Any]) -> str:
    return (
        f"- Video {fact.get('video_id')} @{fact.get('author_username') or 'unknown'} "
        f"(id {fact.get('author_id') or 'unknown'}, url {fact.get('author_url') or 'unavailable'}): "
        f"{fact.get('text')} ({format_metric(fact.get('like_count'))} likes) "
        f"[{comment_citation_label(fact)}]."
    )


def dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for citation in citations:
        label = citation.get("label")
        if label and label not in seen:
            seen.add(label)
            deduped.append(citation)
    return deduped
