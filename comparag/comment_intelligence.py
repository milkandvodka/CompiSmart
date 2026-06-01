from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any

from .analysis_llm import DEFAULT_ANALYSIS_MODEL, gemini_json


JsonDict = dict[str, Any]
WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
PUNCT_RE = re.compile(r"[^\w\s#@]", re.UNICODE)
URL_RE = re.compile(r"https?://|www\.", re.I)


def comment_fingerprint(comments: list[JsonDict]) -> str:
    rows = []
    for comment in comments:
        rows.append(
            {
                "id": comment.get("id") or comment.get("pk") or comment.get("comment_id"),
                "text": comment.get("text"),
                "likes": comment_like_count(comment),
            }
        )
    rows.sort(key=lambda item: str(item.get("id") or item.get("text") or ""))
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def analyze_comments(
    comments: list[JsonDict],
    *,
    video_id: str,
    creator_username: str | None,
    mode: str,
    model: str = DEFAULT_ANALYSIS_MODEL,
    max_evidence_chars: int = 12000,
    cached: JsonDict | None = None,
    force_refresh: bool = False,
) -> JsonDict:
    fingerprint = comment_fingerprint(comments)
    if cached and not force_refresh and cached.get("fingerprint") == fingerprint:
        return {**cached, "cache_hit": True}

    prepared = prepare_comment_evidence(
        comments,
        video_id=video_id,
        creator_username=creator_username,
        max_evidence_chars=max_evidence_chars,
    )
    prepared["fingerprint"] = fingerprint
    prepared["cache_hit"] = False
    prepared["llm_used"] = False
    prepared["model"] = None
    prepared["mode"] = mode

    if mode != "llm":
        if mode == "off":
            return {
                "fingerprint": fingerprint,
                "cache_hit": False,
                "llm_used": False,
                "model": None,
                "mode": "off",
                "available": False,
                "note": "Comment intelligence disabled.",
            }
        prepared["warnings"].append("Comment intelligence requires LLM mode for semantic themes; returning compressed evidence only.")
        return prepared

    try:
        llm_result = llm_comment_intelligence(prepared["evidence_pack"], model=model)
    except Exception as exc:
        prepared["warnings"].append(f"LLM comment intelligence failed; no semantic fallback was generated: {exc}")
        return prepared

    merged = {**prepared, "llm": llm_result}
    merged["themes"] = llm_result.get("themes") or []
    merged["audience_takeaways"] = llm_result.get("audience_takeaways") or []
    merged["recommended_creator_actions"] = llm_result.get("recommended_creator_actions") or []
    merged["objections"] = llm_result.get("objections") or []
    merged["llm_used"] = True
    merged["model"] = model
    return merged


def prepare_comment_evidence(
    comments: list[JsonDict],
    *,
    video_id: str,
    creator_username: str | None,
    max_evidence_chars: int,
) -> JsonDict:
    normalized = [normalize_comment(comment, creator_username=creator_username) for comment in comments if comment.get("text")]
    clusters = build_comment_clusters(normalized)
    noise = Counter(label for comment in normalized for label in comment["labels"] if label in noise_labels())
    signal_comments = [comment for comment in normalized if not set(comment["labels"]).issubset(noise_labels())]
    evidence_pack = build_comment_evidence_pack(
        video_id=video_id,
        total_comments=len(normalized),
        clusters=clusters,
        noise=noise,
        max_chars=max_evidence_chars,
    )
    return {
        "available": True,
        "total_fetched_comments": len(normalized),
        "useful_comment_count": len(signal_comments),
        "cluster_count": len(clusters),
        "compression_ratio": round(len(clusters) / len(normalized), 4) if normalized else 0,
        "noise_summary": dict(noise),
        "top_clusters": clusters[:12],
        "themes": [],
        "audience_takeaways": [],
        "recommended_creator_actions": [],
        "objections": [],
        "evidence_pack": evidence_pack,
        "evidence_pack_chars": len(evidence_pack),
        "warnings": [],
    }


def normalize_comment(comment: JsonDict, *, creator_username: str | None) -> JsonDict:
    text = clean_text(comment.get("text") or "")
    words = WORD_RE.findall(text.lower())
    owner = comment.get("owner") or comment.get("author") or {}
    username = str(owner.get("username") or owner.get("id") or "unknown") if isinstance(owner, dict) else "unknown"
    labels = classify_comment(text, words, username=username, creator_username=creator_username)
    return {
        "id": str(comment.get("id") or comment.get("pk") or comment.get("comment_id") or text[:40]),
        "text": text,
        "normalized_text": normalized_cluster_text(text),
        "words": words,
        "username": username,
        "like_count": comment_like_count(comment),
        "labels": labels,
    }


def classify_comment(text: str, words: list[str], *, username: str, creator_username: str | None) -> list[str]:
    labels = []
    if creator_username and username.lower() == creator_username.lower():
        labels.append("creator_reply")
    if text and not any(char.isalnum() for char in text):
        labels.append("emoji_only")
    if URL_RE.search(text):
        labels.append("contains_url")
    if len(words) <= 2 and not labels:
        labels.append("one_word_low_info")
    if "?" in text:
        labels.append("contains_question_mark")
    if not labels:
        labels.append("unclassified")
    return labels


def build_comment_clusters(comments: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for comment in comments:
        key = comment["normalized_text"] or "emoji_only"
        grouped[key].append(comment)

    clusters = []
    for key, members in grouped.items():
        members.sort(key=lambda item: (item["like_count"], len(item["text"])), reverse=True)
        labels = Counter(label for member in members for label in member["labels"])
        total_likes = sum(member["like_count"] for member in members)
        clusters.append(
            {
                "cluster_key": key[:120],
                "representative_text": members[0]["text"],
                "labels": [label for label, _ in labels.most_common()],
                "count": len(members),
                "total_likes": total_likes,
                "max_likes": max(member["like_count"] for member in members),
                "unique_author_count": len({member["username"] for member in members}),
                "source_comment_ids": [member["id"] for member in members[:25]],
                "examples": [
                    {
                        "text": member["text"],
                        "username": member["username"],
                        "like_count": member["like_count"],
                    }
                    for member in members[:3]
                ],
                "rank_score": round(total_likes + len(members) * 0.75 + len(labels) * 0.1, 4),
            }
        )
    clusters.sort(key=lambda cluster: (cluster["rank_score"], cluster["count"]), reverse=True)
    return clusters


def build_comment_evidence_pack(
    *,
    video_id: str,
    total_comments: int,
    clusters: list[JsonDict],
    noise: Counter[str],
    max_chars: int,
) -> str:
    lines = [
        f"Video {video_id} compressed comment evidence.",
        f"Fetched comments: {total_comments}",
        f"Unique normalized clusters: {len(clusters)}",
        f"Mechanical noise summary: {dict(noise)}",
    ]
    lines.append("")
    lines.append("Top normalized clusters by count and comment-like weight:")
    for cluster in clusters:
        line = (
            f"- labels={','.join(cluster['labels'])}; count={cluster['count']}; "
            f"likes={cluster['total_likes']}; representative={cluster['representative_text']!r}"
        )
        if sum(len(item) + 1 for item in lines) + len(line) > max_chars:
            lines.append("- [truncated: evidence budget reached]")
            break
        lines.append(line)
    return "\n".join(lines)[:max_chars]


def llm_comment_intelligence(evidence_pack: str, *, model: str) -> JsonDict:
    prompt = f"""
Analyze this compressed social-video comment evidence. Return strict JSON only with:
- audience_takeaways: array of 3-6 concise insights
- themes: array of objects with label, description, evidence, comment_count_estimate, importance
- objections: array of objections/questions/confusions
- recommended_creator_actions: array of concrete actions
- notes: array

Rules:
- Use only the compressed evidence.
- Do not treat repeated low-information or emoji-only comments as deep audience meaning.
- Mention when comments are dominated by low-info or campaign-triggered responses.

Evidence:
{evidence_pack}
""".strip()
    return gemini_json(prompt, model=model)


def comment_like_count(comment: JsonDict) -> int:
    value = comment.get("like_count", comment.get("likes_count", 0))
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalized_cluster_text(text: str) -> str:
    lowered = clean_text(text).lower()
    lowered = PUNCT_RE.sub(" ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def noise_labels() -> set[str]:
    return {"creator_reply", "emoji_only", "one_word_low_info"}
