from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import AnalysisConfig, enrich_payload_with_analysis
from .chunking import build_chunks
from .comment_facts import build_comment_fact_table
from .config import get_gemini_api_key, get_openai_api_key
from .config import codex_testing_enabled, env_flag
from .config import supabase_available
from .context import resolve_context_profile
from .embeddings import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODEL_PRESETS, resolve_embedding_model
from .indexing import plan_index_update
from .lexical import LazyBM25LexicalIndex, save_lexical_index
from .llm import (
    CodexTestingLLM,
    ExtractiveFallbackLLM,
    GeminiChatLLM,
    OpenAIChatLLM,
    ProviderFallbackLLM,
    find_codex_command,
)
from .memory import FallbackConversationMemory, InMemoryConversationMemory, SupabaseConversationMemory
from .metrics import build_video_profiles
from .observability import ObservabilityLogger
from .rag_graph import RagChatEngine
from .reranking import DEFAULT_RERANKER_MODEL, RERANKER_MODEL_PRESETS, LocalCrossEncoderReranker, resolve_reranker_model
from .retrieval import HybridRetriever, RetrievalConfig
from .storage import DEFAULT_APP_DIR, load_comparison_record, save_comparison_record
from .vector_store import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION,
    ChromaChunkStore,
    LazyChromaChunkStore,
    collection_name_for_embedding,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index and chat with social-video comparison RAG data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Chunk and embed a normalized extractor JSON file.")
    index_parser.add_argument("--input", "-i", required=True, help="Normalized extractor JSON.")
    index_parser.add_argument("--comparison-id", required=True, help="Stable ID for this A/B comparison.")
    index_parser.add_argument("--app-dir", default=str(DEFAULT_APP_DIR), help="Structured comparison storage directory.")
    index_parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR), help="Local Chroma persistence directory.")
    index_parser.add_argument("--collection", help="Chroma collection name. Defaults to a model-specific collection.")
    index_parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model or preset: {', '.join(EMBEDDING_MODEL_PRESETS)}.",
    )
    index_parser.add_argument("--embedding-device", help="Optional embedding device, e.g. cpu or cuda.")
    index_parser.add_argument(
        "--allow-embedding-download",
        action="store_true",
        help="Allow downloading the embedding model if it is not already cached.",
    )
    index_parser.add_argument(
        "--comment-intelligence",
        choices=["off", "evidence", "llm"],
        default="llm",
        help="Index-time comment intelligence mode. Evidence mode prepares compressed facts without semantic analysis.",
    )
    index_parser.add_argument(
        "--creative-features",
        choices=["off", "evidence", "llm"],
        default="llm",
        help="Index-time transcript evidence mode or LLM-only creative feature extraction.",
    )
    index_parser.add_argument("--analysis-model", default="gemini-2.5-flash-lite", help="Cheap model for analysis calls.")
    index_parser.add_argument(
        "--max-comment-evidence-chars",
        type=int,
        default=12000,
        help="Maximum compressed comment evidence characters sent to the analysis LLM.",
    )
    index_parser.add_argument("--force-analysis-refresh", action="store_true", help="Ignore cached analysis fingerprints.")
    index_parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Delete and re-embed all chunks instead of using the incremental chunk manifest.",
    )
    index_parser.add_argument("--no-reset", action="store_true", help="Do not delete stale chunks during incremental indexing.")

    chat_parser = subparsers.add_parser("chat", help="Ask a question against an indexed comparison.")
    chat_parser.add_argument("--comparison-id", required=True, help="Comparison ID used during indexing.")
    chat_parser.add_argument("--question", "-q", required=True, help="Question to ask.")
    chat_parser.add_argument("--thread-id", default="default", help="Conversation memory thread.")
    chat_parser.add_argument(
        "--memory-backend",
        choices=["auto", "local", "supabase"],
        default="auto",
        help="Conversation memory backend. Auto uses Supabase when env is configured.",
    )
    chat_parser.add_argument("--app-dir", default=str(DEFAULT_APP_DIR), help="Structured comparison storage directory.")
    chat_parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR), help="Local Chroma persistence directory.")
    chat_parser.add_argument("--collection", help="Chroma collection name. Defaults to a model-specific collection.")
    chat_parser.add_argument(
        "--embedding-model",
        help=(
            "Embedding model or preset used for querying. Defaults to the model stored in the "
            f"comparison record. Presets: {', '.join(EMBEDDING_MODEL_PRESETS)}."
        ),
    )
    chat_parser.add_argument("--embedding-device", help="Optional embedding device, e.g. cpu or cuda.")
    chat_parser.add_argument(
        "--allow-embedding-download",
        action="store_true",
        help="Allow downloading the embedding model if it is not already cached.",
    )
    chat_parser.add_argument(
        "--llm",
        choices=["auto", "gemini", "openai", "fallback", "codex_testing"],
        default="auto",
        help="Answer generator. Auto tries Gemini, OpenAI, Codex testing, then retrieval fallback.",
    )
    chat_parser.add_argument("--gemini-model", default="gemini-2.5-flash-lite", help="Gemini model for chat.")
    chat_parser.add_argument(
        "--retrieval-mode",
        choices=["semantic", "lexical", "hybrid"],
        default="hybrid",
        help="Semantic vector, lexical BM25, or hybrid fused retrieval.",
    )
    chat_parser.add_argument("--semantic-weight", type=float, default=1.0, help="Hybrid semantic RRF weight.")
    chat_parser.add_argument("--lexical-weight", type=float, default=1.0, help="Hybrid lexical RRF weight.")
    chat_parser.add_argument("--enable-reranker", action="store_true", help="Rerank retrieved candidates locally.")
    chat_parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help=f"Cross-encoder reranker model or preset: {', '.join(RERANKER_MODEL_PRESETS)}.",
    )
    chat_parser.add_argument("--reranker-device", help="Optional reranker device, e.g. cpu or cuda.")
    chat_parser.add_argument(
        "--allow-reranker-download",
        action="store_true",
        help="Allow downloading the reranker model if it is not already cached.",
    )
    chat_parser.add_argument("--no-stream", action="store_true", help="Print final answer instead of token stream.")

    return parser


def main() -> int:
    configure_console()
    args = build_parser().parse_args()
    if args.command == "index":
        return index_command(args)
    if args.command == "chat":
        return chat_command(args)
    raise ValueError(f"Unknown command: {args.command}")


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def index_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    app_dir = Path(args.app_dir)
    logger = ObservabilityLogger(app_dir=app_dir)
    logger.event("index_start", {"comparison_id": args.comparison_id, "source_path": str(input_path)})
    profiles = build_video_profiles(payload, args.comparison_id)
    previous_record = load_previous_record(args.comparison_id, app_dir)
    enriched_payload, analysis_artifacts = enrich_payload_with_analysis(
        payload,
        profiles=profiles,
        previous_record=previous_record,
        config=AnalysisConfig(
            comment_intelligence=args.comment_intelligence,
            creative_features=args.creative_features,
            analysis_model=args.analysis_model,
            max_comment_evidence_chars=args.max_comment_evidence_chars,
            force_analysis_refresh=args.force_analysis_refresh,
        ),
        logger=logger,
    )
    profiles, chunks = build_chunks(enriched_payload, args.comparison_id)
    comment_facts = build_comment_fact_table(enriched_payload, profiles)
    embedding_model = resolve_embedding_model(args.embedding_model)
    collection_name = resolve_collection_name(args.collection, embedding_model)
    index_plan = plan_index_update(
        chunks,
        previous_record=previous_record,
        force_reindex=args.force_reindex,
        no_delete=args.no_reset,
    )
    vector_write_needed = (
        bool(index_plan.chunks_to_upsert)
        or bool(index_plan.chunk_ids_to_delete)
        or (index_plan.mode in {"force_reindex", "full_no_previous_manifest"} and not args.no_reset)
    )
    if vector_write_needed:
        store = ChromaChunkStore(
            persist_dir=Path(args.chroma_dir),
            collection_name=collection_name,
            embedding_model=embedding_model,
            embedding_device=args.embedding_device,
            allow_embedding_download=args.allow_embedding_download,
        )
        if index_plan.mode in {"force_reindex", "full_no_previous_manifest"} and not args.no_reset:
            store.reset_comparison(args.comparison_id)
        elif index_plan.chunk_ids_to_delete:
            store.delete_chunks(index_plan.chunk_ids_to_delete)
        store.upsert_chunks(index_plan.chunks_to_upsert)
    lexical_path = save_lexical_index(chunks, comparison_id=args.comparison_id, app_dir=app_dir)
    record_path = save_comparison_record(
        comparison_id=args.comparison_id,
        profiles=profiles,
        chunk_count=len(chunks),
        source_path=str(input_path),
        app_dir=app_dir,
        embedding_model=embedding_model,
        collection_name=collection_name,
        analysis_artifacts=analysis_artifacts,
        comment_facts=comment_facts,
        chunk_manifest=index_plan.chunk_manifest,
        indexing_stats=index_plan.stats(),
        observability_run_id=logger.run_id,
    )
    logger.event(
        "index_complete",
        {
            "comparison_id": args.comparison_id,
            "chunk_count": len(chunks),
            "collection_name": collection_name,
            "lexical_path": str(lexical_path),
            "record_path": str(record_path),
            "indexing": index_plan.stats(),
            "vector_write_needed": vector_write_needed,
            "comment_fact_count": len(comment_facts.get("facts") or []),
        },
    )

    print(f"Indexed {len(chunks)} chunks for comparison {args.comparison_id}.")
    print(
        "Index update: "
        f"{index_plan.mode}, upserted {index_plan.upsert_count}, "
        f"deleted {index_plan.delete_count}, unchanged {index_plan.unchanged_count}."
    )
    if not vector_write_needed:
        print("Chroma update skipped: chunk manifest unchanged.")
    print(f"Saved comparison record: {record_path}")
    print(f"Saved lexical index: {lexical_path}")
    print(f"Chroma directory: {Path(args.chroma_dir)}")
    print(f"Chroma collection: {collection_name}")
    print(f"Observability run: {logger.run_id}")
    return 0


def chat_command(args: argparse.Namespace) -> int:
    app_dir = Path(args.app_dir)
    logger = ObservabilityLogger(app_dir=app_dir)
    logger.event(
        "chat_start",
        {
            "comparison_id": args.comparison_id,
            "retrieval_mode": args.retrieval_mode,
            "llm": args.llm,
            "codex_testing_enabled": codex_testing_enabled(),
            "reranker_enabled": args.enable_reranker,
        },
    )
    record = load_comparison_record(args.comparison_id, app_dir)
    embedding_model = resolve_chat_embedding_model(args.embedding_model, record)
    collection_name = resolve_chat_collection_name(args.collection, record, embedding_model)
    semantic_store = LazyChromaChunkStore(
        persist_dir=Path(args.chroma_dir),
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_device=args.embedding_device,
        allow_embedding_download=args.allow_embedding_download,
    )
    lexical_store = LazyBM25LexicalIndex(comparison_id=args.comparison_id, app_dir=app_dir)
    retriever = HybridRetriever(
        semantic_retriever=semantic_store,
        lexical_retriever=lexical_store,
        reranker=build_reranker(args),
        config=RetrievalConfig(
            mode=args.retrieval_mode,
            semantic_weight=args.semantic_weight,
            lexical_weight=args.lexical_weight,
        ),
    )
    llm = build_llm(args)
    engine = RagChatEngine(
        retriever=retriever,
        profiles=record.get("videos") or [],
        comment_facts=record.get("comment_facts") or {},
        context_profile=context_profile_for_args(args),
        llm=llm,
        memory=build_memory(args),
    )

    if args.no_stream:
        result = engine.invoke(comparison_id=args.comparison_id, question=args.question, thread_id=args.thread_id)
        logger.event(
            "chat_complete",
            {
                "route": result.get("route"),
                "evidence_plan": result.get("evidence_plan"),
                "citation_audit": result.get("citation_audit"),
                "citation_count": len(result.get("citations") or []),
                "llm_error": result.get("llm_error"),
            },
        )
        print(result["answer"])
        print("\nCitations:")
        for citation in result.get("citations", []):
            print(f"- {citation['label']}")
        print("\nCitation audit:")
        print(json.dumps(result.get("citation_audit") or {}, ensure_ascii=False))
        return 0

    final_event = None
    for event in engine.stream(comparison_id=args.comparison_id, question=args.question, thread_id=args.thread_id):
        if event["type"] == "metadata":
            print(f"[route: {event.get('route')}]\n", flush=True)
        elif event["type"] == "token":
            print(event["text"], end="", flush=True)
        elif event["type"] == "done":
            final_event = event
            print("\n\nCitations:")
            for citation in event.get("citations", []):
                print(f"- {citation['label']}")
            print("\nCitation audit:")
            print(json.dumps(event.get("citation_audit") or {}, ensure_ascii=False))
    logger.event(
        "chat_complete",
        {
            "citation_audit": (final_event or {}).get("citation_audit"),
            "citation_count": len((final_event or {}).get("citations") or []),
            "llm_error": (final_event or {}).get("llm_error"),
        },
    )
    return 0


def build_llm(args: argparse.Namespace):
    if args.llm == "fallback":
        return ExtractiveFallbackLLM()
    if args.llm == "codex_testing" or codex_testing_enabled():
        return CodexTestingLLM()
    if args.llm == "openai":
        return OpenAIChatLLM()
    if args.llm == "gemini":
        return GeminiChatLLM(model=args.gemini_model)
    providers = []
    if get_gemini_api_key() and not env_flag("COMPARAG_DISABLE_GEMINI"):
        providers.append(("gemini", GeminiChatLLM(model=args.gemini_model)))
    if get_openai_api_key():
        providers.append(("openai", OpenAIChatLLM()))
    if find_codex_command():
        providers.append(("codex_testing", CodexTestingLLM()))
    providers.append(("retrieval_fallback", ExtractiveFallbackLLM()))
    return ProviderFallbackLLM(providers)


def context_profile_for_args(args: argparse.Namespace):
    if args.llm == "codex_testing" or codex_testing_enabled():
        return resolve_context_profile("codex_testing")
    if args.llm in {"gemini", "auto", "openai"}:
        return resolve_context_profile("gemini", args.gemini_model)
    return resolve_context_profile("small")


def build_memory(args: argparse.Namespace):
    if args.memory_backend == "local":
        return InMemoryConversationMemory()
    if args.memory_backend == "supabase":
        return SupabaseConversationMemory()
    if args.memory_backend == "auto" and supabase_available():
        return FallbackConversationMemory(SupabaseConversationMemory())
    return InMemoryConversationMemory()


def load_previous_record(comparison_id: str, app_dir: Path):
    try:
        return load_comparison_record(comparison_id, app_dir)
    except FileNotFoundError:
        return None


def build_reranker(args: argparse.Namespace):
    if not args.enable_reranker:
        return None
    return LocalCrossEncoderReranker(
        model_name=resolve_reranker_model(args.reranker_model),
        device=args.reranker_device,
        allow_download=args.allow_reranker_download,
    )


def resolve_collection_name(collection: str | None, embedding_model: str) -> str:
    if collection:
        return collection
    return collection_name_for_embedding(DEFAULT_COLLECTION, embedding_model)


def resolve_chat_collection_name(
    collection: str | None,
    record: dict[str, object],
    embedding_model: str,
) -> str:
    if collection:
        return collection
    if record.get("embedding_model") == embedding_model and isinstance(record.get("collection_name"), str):
        return str(record["collection_name"])
    return collection_name_for_embedding(DEFAULT_COLLECTION, embedding_model)


def resolve_chat_embedding_model(embedding_model: str | None, record: dict[str, object]) -> str:
    if embedding_model:
        return resolve_embedding_model(embedding_model)
    if isinstance(record.get("embedding_model"), str) and record["embedding_model"]:
        return str(record["embedding_model"])
    return DEFAULT_EMBEDDING_MODEL


if __name__ == "__main__":
    raise SystemExit(main())
