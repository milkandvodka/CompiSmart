from __future__ import annotations

import re
from typing import Any


def build_citations(
    retrieved: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    route: str,
    tool_results: dict[str, Any] | None = None,
    evidence_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    if route in {"metrics", "creator", "comparison", "improvement"} or (evidence_plan or {}).get("needs_structured_metrics"):
        for profile in profiles:
            citations.append(
                {
                    "label": f"Video {profile.get('video_id')}, metadata snapshot",
                    "video_id": profile.get("video_id"),
                    "doc_type": "video_fact_card",
                }
            )
    for citation in (tool_results or {}).get("citations") or []:
        citations.append(dict(citation))
    for chunk in retrieved:
        metadata = chunk.get("metadata") or {}
        label = metadata.get("citation_label") or chunk.get("id")
        citations.append(
            {
                "label": label,
                "video_id": metadata.get("video_id"),
                "doc_type": metadata.get("doc_type"),
                "chunk_id": chunk.get("id"),
            }
        )
    return dedupe_citations(citations)


def dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for citation in citations:
        label = str(citation.get("label") or "")
        if not label or label in seen:
            continue
        seen.add(label)
        deduped.append(citation)
    return deduped


def validate_answer_citations(answer: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = {str(citation.get("label")) for citation in citations if citation.get("label")}
    cited = sorted(
        {
            label
            for match in re.findall(r"\[([^\]]+)\]", answer or "")
            for label in split_citation_label_match(match)
        }
    )
    invalid = [label for label in cited if label not in allowed]
    return {
        "allowed_labels": sorted(allowed),
        "cited_labels": cited,
        "invalid_labels": invalid,
        "valid": not invalid,
    }


def split_citation_label_match(match: str) -> list[str]:
    text = match.strip()
    if not text.startswith("Video "):
        return []
    return [part.strip() for part in re.split(r",\s+(?=Video\s+[A-Z]\b)", text) if part.strip()]
