from __future__ import annotations

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
    if evidence_plan.get("video_id"):
        return retriever.query(
            question,
            comparison_id=comparison_id,
            video_id=str(evidence_plan["video_id"]),
            doc_types=list(evidence_plan["doc_types"]),
            n_results=int(evidence_plan.get("n_results") or 8),
        )
    if evidence_plan.get("balanced_retrieval"):
        return balanced_retrieve(
            retriever,
            question,
            comparison_id=comparison_id,
            doc_types=list(evidence_plan["doc_types"]),
            per_video=int(evidence_plan.get("per_video") or 4),
        )
    return retriever.query(
        question,
        comparison_id=comparison_id,
        doc_types=list(evidence_plan["doc_types"]),
        n_results=int(evidence_plan.get("n_results") or 8),
    )


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
