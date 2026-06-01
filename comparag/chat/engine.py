from __future__ import annotations

from typing import Any, Iterable

try:
    from langgraph.graph import END, StateGraph  # type: ignore
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
    END = None
    StateGraph = None
    LANGGRAPH_IMPORT_ERROR = exc
else:
    LANGGRAPH_IMPORT_ERROR = None

from ..comment_facts import query_comment_facts
from ..context import DEFAULT_CONTEXT_PROFILE, ModelContextProfile
from ..llm import ChatLLM, ExtractiveFallbackLLM
from ..memory import ConversationMemory, InMemoryConversationMemory
from ..models import RetrievedChunk
from .citations import build_citations, validate_answer_citations
from .prompting import build_memory_summary_prompt, build_prompt, generation_failure_answer
from .retrieval import retrieve_for_evidence_plan
from .routing import fallback_evidence_plan, plan_question_with_llm
from .state import ChatState, ChunkRetriever
from .utils import as_int, fit_to_budget


class RagChatEngine:
    def __init__(
        self,
        *,
        retriever: ChunkRetriever,
        profiles: list[dict[str, Any]],
        comment_facts: dict[str, Any] | None = None,
        context_profile: ModelContextProfile = DEFAULT_CONTEXT_PROFILE,
        llm: ChatLLM | None = None,
        memory: ConversationMemory | None = None,
    ):
        self.retriever = retriever
        self.profiles = profiles
        self.comment_facts = comment_facts or {}
        self.context_profile = context_profile
        self.llm = llm or ExtractiveFallbackLLM()
        self.memory = memory or InMemoryConversationMemory()
        self.graph = build_prepare_graph(
            retriever=retriever,
            profiles=profiles,
            comment_facts=self.comment_facts,
            context_profile=self.context_profile,
            planner_llm=self.llm,
        )

    def invoke(self, *, comparison_id: str, question: str, thread_id: str = "default") -> dict[str, Any]:
        history = self.memory.get(thread_id)
        summary_record = self.memory.get_summary(thread_id)
        prepared = self.graph.invoke(
            {
                "comparison_id": comparison_id,
                "question": question,
                "history": history,
                "memory_summary": str(summary_record.get("summary") or ""),
                "memory_summary_metadata": summary_record.get("metadata") or {},
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        llm_error = None
        try:
            answer = self.llm.complete(prepared["prompt"])
        except Exception as exc:
            llm_error = f"{exc.__class__.__name__}: generation failed"
            answer = generation_failure_answer(prepared, llm_error)
        citation_audit = validate_answer_citations(answer, prepared.get("citations", []))
        self.memory.append_turn(thread_id, user=question, assistant=answer)
        summary_update = self.maybe_update_memory_summary(thread_id)
        return {
            "answer": answer,
            "citations": prepared.get("citations", []),
            "citation_audit": citation_audit,
            "evidence_plan": prepared.get("evidence_plan"),
            "route": prepared.get("route"),
            "llm_error": llm_error,
            "memory_summary_update": summary_update,
        }

    def stream(self, *, comparison_id: str, question: str, thread_id: str = "default") -> Iterable[dict[str, Any]]:
        history = self.memory.get(thread_id)
        summary_record = self.memory.get_summary(thread_id)
        prepared = self.graph.invoke(
            {
                "comparison_id": comparison_id,
                "question": question,
                "history": history,
                "memory_summary": str(summary_record.get("summary") or ""),
                "memory_summary_metadata": summary_record.get("metadata") or {},
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        citations = prepared.get("citations", [])
        yield {
            "type": "metadata",
            "route": prepared.get("route"),
            "evidence_plan": prepared.get("evidence_plan"),
            "citations": citations,
        }
        answer_parts: list[str] = []
        llm_error = None
        try:
            stream_source = self.llm.stream(prepared["prompt"])
            for token in stream_source:
                answer_parts.append(token)
                yield {"type": "token", "text": token}
        except Exception as exc:
            llm_error = f"{exc.__class__.__name__}: generation failed"
            fallback_answer = generation_failure_answer(prepared, llm_error)
            answer_parts.append(fallback_answer)
            yield {"type": "token", "text": fallback_answer}
        answer = "".join(answer_parts)
        citation_audit = validate_answer_citations(answer, citations)
        self.memory.append_turn(thread_id, user=question, assistant=answer)
        summary_update = self.maybe_update_memory_summary(thread_id)
        yield {
            "type": "done",
            "citations": citations,
            "citation_audit": citation_audit,
            "llm_error": llm_error,
            "memory_summary_update": summary_update,
        }

    def maybe_update_memory_summary(self, thread_id: str) -> dict[str, Any]:
        if isinstance(self.llm, ExtractiveFallbackLLM):
            return {"updated": False, "reason": "No LLM summarizer configured."}
        current_count = self.memory.message_count(thread_id)
        summary_record = self.memory.get_summary(thread_id)
        metadata = summary_record.get("metadata") if isinstance(summary_record.get("metadata"), dict) else {}
        last_count = as_int(metadata.get("message_count"))
        delta = current_count - last_count
        if delta < self.context_profile.memory_summary_trigger_messages:
            return {"updated": False, "message_count": current_count, "messages_since_summary": delta}
        recent_messages = self.memory.get(thread_id)[-self.context_profile.memory_recent_messages_for_summary :]
        prompt = build_memory_summary_prompt(
            existing_summary=str(summary_record.get("summary") or ""),
            recent_messages=recent_messages,
            max_chars=self.context_profile.max_memory_summary_chars,
        )
        try:
            summary = fit_to_budget(self.llm.complete(prompt).strip(), self.context_profile.max_memory_summary_chars)
            if not summary:
                return {"updated": False, "message_count": current_count, "reason": "Summarizer returned empty text."}
            self.memory.save_summary(
                thread_id,
                summary,
                {
                    "message_count": current_count,
                    "summary_model": self.llm.__class__.__name__,
                    "recent_message_count": len(recent_messages),
                },
            )
            return {"updated": True, "message_count": current_count, "summary_chars": len(summary)}
        except Exception as exc:
            return {"updated": False, "message_count": current_count, "error": f"{exc.__class__.__name__}: summary failed"}


def build_prepare_graph(
    *,
    retriever: ChunkRetriever,
    profiles: list[dict[str, Any]],
    comment_facts: dict[str, Any] | None = None,
    context_profile: ModelContextProfile = DEFAULT_CONTEXT_PROFILE,
    planner_llm: ChatLLM | None = None,
) -> Any:
    if StateGraph is None or END is None:
        raise RuntimeError("Missing dependency: langgraph. Install with: pip install -r requirements.txt") from LANGGRAPH_IMPORT_ERROR

    def route_node(state: ChatState) -> ChatState:
        plan = plan_question_with_llm(
            question=state["question"],
            history=state.get("history") or [],
            memory_summary=state.get("memory_summary") or "",
            profiles=profiles,
            llm=planner_llm,
            context_profile=context_profile,
        )
        return {
            "route": plan["route"],
            "evidence_plan": plan,
            "profiles": profiles,
            "context_profile": context_profile,
        }

    def evidence_plan_node(state: ChatState) -> ChatState:
        return {"evidence_plan": state.get("evidence_plan") or fallback_evidence_plan(state["question"], state.get("history") or [])}

    def tool_node(state: ChatState) -> ChatState:
        if (state.get("evidence_plan") or {}).get("use_comment_fact_tool"):
            return {"tool_results": query_comment_facts(state["question"], comment_facts or {}, history=state.get("history") or [])}
        return {"tool_results": {}}

    def retrieve_node(state: ChatState) -> ChatState:
        if (state.get("evidence_plan") or {}).get("tool_only") and (state.get("tool_results") or {}).get("available"):
            return {"retrieved": []}
        retrieved = retrieve_for_evidence_plan(
            retriever=retriever,
            comparison_id=state["comparison_id"],
            question=state["question"],
            evidence_plan=state["evidence_plan"],
        )
        return {"retrieved": [chunk_to_dict(chunk) for chunk in retrieved]}

    def prompt_node(state: ChatState) -> ChatState:
        citations = build_citations(
            state.get("retrieved", []),
            state.get("profiles", []),
            state["route"],
            state.get("tool_results", {}),
            state.get("evidence_plan", {}),
        )
        return {"prompt": build_prompt(state, citations), "citations": citations}

    graph = StateGraph(ChatState)
    graph.add_node("route", route_node)
    graph.add_node("evidence_plan", evidence_plan_node)
    graph.add_node("tools", tool_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("prompt", prompt_node)
    graph.set_entry_point("route")
    graph.add_edge("route", "evidence_plan")
    graph.add_edge("evidence_plan", "tools")
    graph.add_edge("tools", "retrieve")
    graph.add_edge("retrieve", "prompt")
    graph.add_edge("prompt", END)
    return graph.compile()


def chunk_to_dict(chunk: RetrievedChunk) -> dict[str, Any]:
    return {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata, "distance": chunk.distance}
