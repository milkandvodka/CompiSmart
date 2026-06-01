from __future__ import annotations

from typing import Any

from ..metrics import format_metric
from .formatting import comment_citation, dedupe_citations, format_comment_fact_line
from .parsing import (
    asks_for_single_top_user,
    asks_most_liked_commenter,
    asks_top_comments,
    extract_comment_ids_from_history,
    extract_comment_phrases,
    extract_comment_phrases_from_history,
    phrase_tasks,
    requested_video_from_text,
    should_map_unscoped_phrases_to_platforms,
    should_resolve_comment_phrase_from_history,
)
from .utils import normalize_comment_text


def query_comment_facts(
    question: str,
    comment_facts: dict[str, Any] | None,
    *,
    max_rows: int = 12,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    facts = list((comment_facts or {}).get("facts") or [])
    if not facts:
        return {"available": False, "answer_text": "", "citations": [], "facts": [], "note": "No structured comments available."}

    text = question.lower()
    requested_video = requested_video_from_text(text)
    phrases = extract_comment_phrases(question)
    if not phrases and should_resolve_comment_phrase_from_history(text):
        comment_ids = extract_comment_ids_from_history(history or [])
        if comment_ids:
            return query_comment_id_matches(facts, comment_ids, requested_video=requested_video, max_rows=max_rows)
        phrases = extract_comment_phrases_from_history(history or [])
    if phrases:
        return query_phrase_matches(
            facts,
            phrases,
            requested_video=requested_video,
            max_rows=max_rows,
            map_unscoped_phrases=should_map_unscoped_phrases_to_platforms(question, phrases),
        )
    if asks_most_liked_commenter(text):
        row_limit = 1 if asks_for_single_top_user(text) else max_rows
        return query_most_liked_commenters(facts, requested_video=requested_video, max_rows=row_limit)
    if asks_top_comments(text):
        return query_top_comment_facts(facts, requested_video=requested_video, max_rows=max_rows)
    return {"available": False, "answer_text": "", "citations": [], "facts": [], "note": "No exact comment-fact tool matched."}


def query_phrase_matches(
    facts: list[dict[str, Any]],
    phrases: list[str],
    *,
    requested_video: str | None,
    max_rows: int,
    map_unscoped_phrases: bool = False,
) -> dict[str, Any]:
    tasks = phrase_tasks(phrases, requested_video, map_unscoped_phrases=map_unscoped_phrases)
    lines = []
    result_rows = []
    citations = []
    for task in tasks:
        video_id = task["video_id"]
        phrase = task["phrase"]
        normalized_phrase = normalize_comment_text(phrase)
        matches = [
            fact
            for fact in facts
            if (not video_id or fact.get("video_id") == video_id)
            and normalized_phrase
            and normalized_phrase in str(fact.get("normalized_text") or "")
        ]
        total_likes = sum(int(fact.get("like_count") or 0) for fact in matches)
        lines.append(
            f"Phrase {phrase!r}"
            + (f" in Video {video_id}" if video_id else "")
            + f": {len(matches)} matching comments, {format_metric(total_likes)} total comment likes."
        )
        for fact in sorted(matches, key=lambda item: int(item.get("like_count") or 0), reverse=True)[:max_rows]:
            result_rows.append(fact)
            citations.append(comment_citation(fact))
            lines.append(format_comment_fact_line(fact))
    return {
        "available": True,
        "tool": "comment_phrase_match",
        "answer_text": "\n".join(lines),
        "facts": result_rows,
        "citations": dedupe_citations(citations),
    }


def query_most_liked_commenters(
    facts: list[dict[str, Any]],
    *,
    requested_video: str | None,
    max_rows: int,
) -> dict[str, Any]:
    scoped = [fact for fact in facts if not requested_video or fact.get("video_id") == requested_video]
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for fact in scoped:
        key = (
            str(fact.get("video_id") or ""),
            str(fact.get("author_id") or ""),
            str(fact.get("author_username") or "unknown"),
        )
        item = grouped.setdefault(
            key,
            {
                "video_id": fact.get("video_id"),
                "author_username": fact.get("author_username"),
                "author_id": fact.get("author_id"),
                "author_url": fact.get("author_url"),
                "comment_count": 0,
                "total_comment_likes": 0,
                "max_comment_likes": 0,
                "top_comment": "",
                "top_comment_id": "",
            },
        )
        likes = int(fact.get("like_count") or 0)
        item["comment_count"] += 1
        item["total_comment_likes"] += likes
        if likes >= int(item.get("max_comment_likes") or 0):
            item["max_comment_likes"] = likes
            item["top_comment"] = fact.get("text")
            item["top_comment_id"] = fact.get("comment_id")
    rows = sorted(grouped.values(), key=lambda item: (item["total_comment_likes"], item["max_comment_likes"]), reverse=True)[
        :max_rows
    ]
    lines = ["Most-liked commenters by total fetched comment likes:"]
    citations = []
    for row in rows:
        label = f"Video {row.get('video_id')}, commenter {row.get('author_username')}"
        citations.append({"label": label, "video_id": row.get("video_id"), "doc_type": "comment_fact"})
        lines.append(
            f"- Video {row.get('video_id')} @{row.get('author_username')}: "
            f"{row.get('comment_count')} comments, {format_metric(row.get('total_comment_likes'))} total likes, "
            f"user id {row.get('author_id') or 'unknown'}, url {row.get('author_url') or 'unavailable'} [{label}]."
        )
    return {"available": True, "tool": "comment_top_commenters", "answer_text": "\n".join(lines), "facts": rows, "citations": citations}


def query_top_comment_facts(
    facts: list[dict[str, Any]],
    *,
    requested_video: str | None,
    max_rows: int,
) -> dict[str, Any]:
    scoped = [fact for fact in facts if not requested_video or fact.get("video_id") == requested_video]
    rows = sorted(scoped, key=lambda fact: int(fact.get("like_count") or 0), reverse=True)[:max_rows]
    lines = ["Top fetched comments by comment-like count:"]
    citations = []
    for fact in rows:
        citations.append(comment_citation(fact))
        lines.append(format_comment_fact_line(fact))
    return {"available": True, "tool": "comment_top_comments", "answer_text": "\n".join(lines), "facts": rows, "citations": citations}


def query_comment_id_matches(
    facts: list[dict[str, Any]],
    comment_ids: list[str],
    *,
    requested_video: str | None,
    max_rows: int,
) -> dict[str, Any]:
    wanted = set(comment_ids)
    rows = [
        fact
        for fact in facts
        if str(fact.get("comment_id") or "") in wanted
        and (not requested_video or fact.get("video_id") == requested_video)
    ][:max_rows]
    citations = [comment_citation(fact) for fact in rows]
    lines = ["Commenters from previous exact comment citations:"]
    for fact in rows:
        lines.append(format_comment_fact_line(fact))
    return {
        "available": True,
        "tool": "comment_id_followup",
        "answer_text": "\n".join(lines),
        "facts": rows,
        "citations": dedupe_citations(citations),
    }
