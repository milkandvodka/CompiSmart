from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import RetrievedChunk


class Retriever(Protocol):
    def query(
        self,
        query_text: str,
        *,
        comparison_id: str,
        video_id: str | None = None,
        doc_types: list[str] | None = None,
        n_results: int = 6,
    ) -> list[RetrievedChunk]:
        ...


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], *, top_n: int) -> list[RetrievedChunk]:
        ...


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "hybrid"
    semantic_pool_multiplier: int = 4
    lexical_pool_multiplier: int = 4
    fusion_k: int = 60
    semantic_weight: float = 1.0
    lexical_weight: float = 1.0
    rerank_top_n: int = 20


class HybridRetriever:
    def __init__(
        self,
        *,
        semantic_retriever: Retriever,
        lexical_retriever: Retriever,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ):
        self.semantic_retriever = semantic_retriever
        self.lexical_retriever = lexical_retriever
        self.reranker = reranker
        self.config = config or RetrievalConfig()

    def query(
        self,
        query_text: str,
        *,
        comparison_id: str,
        video_id: str | None = None,
        doc_types: list[str] | None = None,
        n_results: int = 6,
    ) -> list[RetrievedChunk]:
        mode = self.config.mode
        if mode == "semantic":
            chunks = self.semantic_retriever.query(
                query_text,
                comparison_id=comparison_id,
                video_id=video_id,
                doc_types=doc_types,
                n_results=max(n_results, self.config.rerank_top_n if self.reranker else n_results),
            )
            return self.apply_reranker(query_text, mark_source(chunks, "semantic"), n_results)
        if mode == "lexical":
            chunks = self.lexical_retriever.query(
                query_text,
                comparison_id=comparison_id,
                video_id=video_id,
                doc_types=doc_types,
                n_results=max(n_results, self.config.rerank_top_n if self.reranker else n_results),
            )
            return self.apply_reranker(query_text, mark_source(chunks, "lexical"), n_results)
        if mode != "hybrid":
            raise ValueError(f"Unknown retrieval mode: {mode}")

        semantic_pool = max(n_results, n_results * self.config.semantic_pool_multiplier)
        lexical_pool = max(n_results, n_results * self.config.lexical_pool_multiplier)
        semantic_chunks = self.semantic_retriever.query(
            query_text,
            comparison_id=comparison_id,
            video_id=video_id,
            doc_types=doc_types,
            n_results=semantic_pool,
        )
        lexical_chunks = self.lexical_retriever.query(
            query_text,
            comparison_id=comparison_id,
            video_id=video_id,
            doc_types=doc_types,
            n_results=lexical_pool,
        )
        fused = reciprocal_rank_fusion(
            semantic_chunks=semantic_chunks,
            lexical_chunks=lexical_chunks,
            fusion_k=self.config.fusion_k,
            semantic_weight=self.config.semantic_weight,
            lexical_weight=self.config.lexical_weight,
        )
        candidate_count = max(n_results, self.config.rerank_top_n if self.reranker else n_results)
        return self.apply_reranker(query_text, fused[:candidate_count], n_results)

    def apply_reranker(self, query_text: str, chunks: list[RetrievedChunk], n_results: int) -> list[RetrievedChunk]:
        if not self.reranker:
            return chunks[:n_results]
        return self.reranker.rerank(query_text, chunks, top_n=n_results)


def reciprocal_rank_fusion(
    *,
    semantic_chunks: list[RetrievedChunk],
    lexical_chunks: list[RetrievedChunk],
    fusion_k: int,
    semantic_weight: float,
    lexical_weight: float,
) -> list[RetrievedChunk]:
    candidates: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    semantic_ranks: dict[str, int] = {}
    lexical_ranks: dict[str, int] = {}

    for rank, chunk in enumerate(semantic_chunks, start=1):
        candidates.setdefault(chunk.id, chunk)
        semantic_ranks[chunk.id] = rank
        scores[chunk.id] = scores.get(chunk.id, 0.0) + semantic_weight / (fusion_k + rank)

    for rank, chunk in enumerate(lexical_chunks, start=1):
        candidates.setdefault(chunk.id, chunk)
        lexical_ranks[chunk.id] = rank
        scores[chunk.id] = scores.get(chunk.id, 0.0) + lexical_weight / (fusion_k + rank)

    ordered_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    fused: list[RetrievedChunk] = []
    for chunk_id in ordered_ids:
        chunk = candidates[chunk_id]
        source = source_label(chunk_id, semantic_ranks, lexical_ranks)
        fused.append(
            RetrievedChunk(
                id=chunk.id,
                text=chunk.text,
                metadata={
                    **chunk.metadata,
                    "retrieval_source": source,
                    "semantic_rank": semantic_ranks.get(chunk_id),
                    "lexical_rank": lexical_ranks.get(chunk_id),
                    "fusion_score": round(scores[chunk_id], 8),
                },
                distance=chunk.distance,
            )
        )
    return fused


def mark_source(chunks: list[RetrievedChunk], source: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            id=chunk.id,
            text=chunk.text,
            metadata={**chunk.metadata, "retrieval_source": source},
            distance=chunk.distance,
        )
        for chunk in chunks
    ]


def source_label(chunk_id: str, semantic_ranks: dict[str, int], lexical_ranks: dict[str, int]) -> str:
    in_semantic = chunk_id in semantic_ranks
    in_lexical = chunk_id in lexical_ranks
    if in_semantic and in_lexical:
        return "hybrid"
    if in_semantic:
        return "semantic"
    return "lexical"
