from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparag.cli import resolve_chat_collection_name, resolve_chat_embedding_model
from comparag.context import resolve_context_profile
from comparag.lexical import LazyBM25LexicalIndex
from comparag.llm import CodexTestingLLM
from comparag.memory import InMemoryConversationMemory
from comparag.rag_graph import RagChatEngine
from comparag.retrieval import HybridRetriever, RetrievalConfig
from comparag.storage import DEFAULT_APP_DIR, load_comparison_record
from comparag.vector_store import DEFAULT_CHROMA_DIR, LazyChromaChunkStore


QUESTIONS = [
    "Start by giving me the core metrics for both videos: views, likes, comments, engagement rate, duration, upload date, creator, follower count, and thumbnails.",
    "Which video performed better on engagement rate, and how different is it from the other one?",
    "Compare the hooks in the first 5 seconds. Which one sounds catchier and why?",
    "Who is the creator of Video B, what is their follower count, and what is the thumbnail URL?",
    "Show me the top Instagram comments you fetched with usernames, comment-like counts, user ids, and profile URLs.",
    "Who commented 'link' in the Instagram video? Give count and total comment likes.",
    "What were their total comment likes again, and which user had the most among them?",
    "What revenue, sales, conversion rate, watch time, and retention did each video get?",
    "What is the capital of Japan, and can you tie that to why Video A performed better?",
    "Give me the profile of the fetched commenter with the most comment likes across both videos.",
    "Now summarize the whole comparison and suggest improvements for Video B based only on evidence.",
]


def main() -> int:
    configure_console()
    load_dotenv(Path(".env"))
    comparison_id = os.environ.get("COMPARAG_BATTERY_COMPARISON_ID", "qa_llm")
    app_dir = Path(os.environ.get("COMPARAG_APP_DIR", str(DEFAULT_APP_DIR)))
    chroma_dir = Path(os.environ.get("COMPARAG_CHROMA_DIR", str(DEFAULT_CHROMA_DIR)))
    output_path = Path(os.environ.get("COMPARAG_BATTERY_OUTPUT", ".cache/final_agent_conversation_battery.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record = load_comparison_record(comparison_id, app_dir)
    embedding_model = resolve_chat_embedding_model(None, record)
    collection_name = resolve_chat_collection_name(None, record, embedding_model)
    semantic_store = LazyChromaChunkStore(
        persist_dir=chroma_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        allow_embedding_download=False,
    )
    lexical_store = LazyBM25LexicalIndex(comparison_id=comparison_id, app_dir=app_dir)
    retriever = HybridRetriever(
        semantic_retriever=semantic_store,
        lexical_retriever=lexical_store,
        config=RetrievalConfig(mode="hybrid"),
    )
    memory = InMemoryConversationMemory()
    llm = CodexTestingLLM(timeout_seconds=180)
    engine = RagChatEngine(
        retriever=retriever,
        profiles=record.get("videos") or [],
        comment_facts=record.get("comment_facts") or {},
        context_profile=resolve_context_profile("codex_testing"),
        llm=llm,
        memory=memory,
    )

    thread_id = f"final_battery_{uuid4().hex[:8]}"
    results = []
    started = time.perf_counter()
    print(f"thread_id={thread_id}")
    print(f"comparison_id={comparison_id}")
    questions = load_questions()
    print(f"questions={len(questions)}")

    for index, question in enumerate(questions, 1):
        turn_start = time.perf_counter()
        print(f"\n--- TURN {index} ---")
        print(f"Q: {question}")
        try:
            result = engine.invoke(comparison_id=comparison_id, question=question, thread_id=thread_id)
        except Exception as exc:
            result = {
                "answer": "",
                "citations": [],
                "citation_audit": {"valid": False, "error": f"{exc.__class__.__name__}: {exc}"},
                "evidence_plan": {},
                "route": None,
                "llm_error": f"{exc.__class__.__name__}: invoke failed",
                "memory_summary_update": {},
            }
        elapsed = round(time.perf_counter() - turn_start, 2)
        preview = compact(result.get("answer") or "", 1200)
        print(f"route={result.get('route')} elapsed={elapsed}s")
        print(f"citation_valid={(result.get('citation_audit') or {}).get('valid')}")
        print(f"summary_update={result.get('memory_summary_update')}")
        print("A:")
        print(preview)
        results.append(
            {
                "turn": index,
                "question": question,
                "elapsed_seconds": elapsed,
                "route": result.get("route"),
                "answer": result.get("answer"),
                "citations": result.get("citations") or [],
                "citation_audit": result.get("citation_audit") or {},
                "evidence_plan": result.get("evidence_plan") or {},
                "llm_error": result.get("llm_error"),
                "memory_summary_update": result.get("memory_summary_update") or {},
            }
        )
        write_checkpoint(
            output_path,
            {
                "thread_id": thread_id,
                "comparison_id": comparison_id,
                "turns": results,
                "message_count": memory.message_count(thread_id),
                "final_memory_summary": memory.get_summary(thread_id),
                "partial": True,
            },
        )

    summary_record = memory.get_summary(thread_id)
    payload = {
        "thread_id": thread_id,
        "comparison_id": comparison_id,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "turns": results,
        "final_memory_summary": summary_record,
        "message_count": memory.message_count(thread_id),
    }
    write_checkpoint(output_path, payload)
    print(f"\noutput={output_path}")
    print(f"total_elapsed_seconds={payload['elapsed_seconds']}")
    print(f"message_count={payload['message_count']}")
    print(f"final_summary_chars={len(str(summary_record.get('summary') or ''))}")
    return 0


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_questions() -> list[str]:
    raw = os.environ.get("COMPARAG_BATTERY_QUESTIONS_JSON")
    if not raw:
        return QUESTIONS
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("COMPARAG_BATTERY_QUESTIONS_JSON must be a JSON array of strings.")
    return parsed


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def write_checkpoint(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def compact(text: str, max_chars: int) -> str:
    normalized = " ".join(str(text or "").split())
    return normalized if len(normalized) <= max_chars else normalized[: max_chars - 1].rstrip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())
