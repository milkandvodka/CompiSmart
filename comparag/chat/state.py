from __future__ import annotations

from typing import Any, Protocol, TypedDict

from ..memory import Message
from ..models import RetrievedChunk
from ..context import ModelContextProfile


class ChunkRetriever(Protocol):
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


class ChatState(TypedDict, total=False):
    comparison_id: str
    question: str
    history: list[Message]
    memory_summary: str
    memory_summary_metadata: dict[str, Any]
    route: str
    evidence_plan: dict[str, Any]
    tool_results: dict[str, Any]
    context_profile: ModelContextProfile
    profiles: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    citation_audit: dict[str, Any]
    prompt: str
    planner_error: str
