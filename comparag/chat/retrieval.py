from __future__ import annotations

import re

from ..models import RetrievedChunk
from .routing import fallback_evidence_plan_for_route
from .state import ChunkRetriever


def retrieve_for_evidence_plan(
    *,
    retriever: ChunkRetriever,
    comparison_id: str,
    question: str,
    evidence_plan: dict[str, Any],
) -> list[RetrievedChunk]:
    if not evidence_plan.get("doc_types"):
        return []
    questions = retrieval_questions(question)
    if evidence_plan.get("video_id"):
        chunks: list[RetrievedChunk] = []
        for query in questions:
            chunks.extend(
                retriever.query(
                    query,
                    comparison_id=comparison_id,
                    video_id=str(evidence_plan["video_id"]),
                    doc_types=list(evidence_plan["doc_types"]),
                    n_results=int(evidence_plan.get("n_results") or 8),
                )
            )
        return limit_retrieved(dedupe_retrieved(chunks), int(evidence_plan.get("n_results") or 8))
    if evidence_plan.get("balanced_retrieval"):
        chunks = []
        for query in questions:
            chunks.extend(
                balanced_retrieve(
                    retriever,
                    query,
                    comparison_id=comparison_id,
                    doc_types=list(evidence_plan["doc_types"]),
                    per_video=int(evidence_plan.get("per_video") or 4),
                )
            )
        return limit_retrieved(dedupe_retrieved(chunks), max(8, int(evidence_plan.get("per_video") or 4) * 2))
    chunks = []
    for query in questions:
        chunks.extend(
            retriever.query(
                query,
                comparison_id=comparison_id,
                doc_types=list(evidence_plan["doc_types"]),
                n_results=int(evidence_plan.get("n_results") or 8),
            )
        )
    return limit_retrieved(dedupe_retrieved(chunks), max(8, int(evidence_plan.get("n_results") or 8)))


def retrieve_for_route(
    *,
    retriever: ChunkRetriever,
    comparison_id: str,
    question: str,
    route: str,
) -> list[RetrievedChunk]:
    return retrieve_for_evidence_plan(
        retriever=retriever,
        comparison_id=comparison_id,
        question=question,
        evidence_plan=fallback_evidence_plan_for_route(question, route),
    )


def balanced_retrieve(
    retriever: ChunkRetriever,
    question: str,
    *,
    comparison_id: str,
    doc_types: list[str],
    per_video: int,
) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for video_id in ("A", "B"):
        chunks.extend(
            retriever.query(
                question,
                comparison_id=comparison_id,
                video_id=video_id,
                doc_types=doc_types,
                n_results=per_video,
            )
        )
    return dedupe_retrieved(chunks)


def dedupe_retrieved(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    deduped: list[RetrievedChunk] = []
    for chunk in chunks:
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        deduped.append(chunk)
    return deduped


def limit_retrieved(chunks: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
    return chunks[: max(1, min(24, limit))]


def retrieval_questions(question: str) -> list[str]:
    cleaned = " ".join(str(question or "").split())
    if not cleaned:
        return []
    pieces = [
        piece.strip(" .?!,;:")
        for piece in re.split(r"\b(?:and also|also|plus|along with)\b|[?]\s+", cleaned, flags=re.I)
        if piece.strip(" .?!,;:")
    ]
    queries = [cleaned]
    for piece in pieces:
        if len(piece.split()) >= 3 and piece.lower() not in {query.lower() for query in queries}:
            queries.append(piece)
    return queries[:5]
