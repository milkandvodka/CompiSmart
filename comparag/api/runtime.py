from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import codex_testing_enabled, env_flag, get_gemini_api_key, get_openai_api_key, supabase_available
from ..context import resolve_context_profile
from ..embeddings import DEFAULT_EMBEDDING_MODEL, resolve_embedding_model
from ..lexical import LazyBM25LexicalIndex
from ..llm import CodexTestingLLM, ExtractiveFallbackLLM, GeminiChatLLM, OpenAIChatLLM, ProviderFallbackLLM, find_codex_command
from ..memory import FallbackConversationMemory, InMemoryConversationMemory, SupabaseConversationMemory
from ..rag_graph import RagChatEngine
from ..reranking import LocalCrossEncoderReranker, resolve_reranker_model
from ..retrieval import HybridRetriever, RetrievalConfig
from ..vector_store import DEFAULT_COLLECTION, LazyChromaChunkStore, collection_name_for_embedding
from .paths import resolve_chroma_dir
from .schemas import AgentRuntimeOptions


def build_engine(
    *,
    comparison_id: str,
    record: dict[str, Any],
    options: AgentRuntimeOptions,
    app_dir: Path,
    embedding_model: str,
    collection_name: str,
) -> RagChatEngine:
    semantic_store = LazyChromaChunkStore(
        persist_dir=resolve_chroma_dir(options.chroma_dir),
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_device=options.embedding_device,
        allow_embedding_download=options.allow_embedding_download,
    )
    lexical_store = LazyBM25LexicalIndex(comparison_id=comparison_id, app_dir=app_dir)
    retriever = HybridRetriever(
        semantic_retriever=semantic_store,
        lexical_retriever=lexical_store,
        reranker=build_reranker(options),
        config=RetrievalConfig(
            mode=options.retrieval_mode,
            semantic_weight=options.semantic_weight,
            lexical_weight=options.lexical_weight,
        ),
    )
    llm = build_llm(options)
    return RagChatEngine(
        retriever=retriever,
        profiles=record.get("videos") or [],
        comment_facts=record.get("comment_facts") or {},
        context_profile=context_profile_for_options(options),
        llm=llm,
        memory=build_memory(options),
    )


def build_llm(options: AgentRuntimeOptions):
    mode = resolve_llm_mode(options.llm)
    if mode == "fallback":
        return ExtractiveFallbackLLM()
    if mode == "codex_testing":
        return CodexTestingLLM()
    if mode == "openai":
        return OpenAIChatLLM()
    if mode == "gemini":
        return GeminiChatLLM(model=options.gemini_model)
    return build_auto_llm(options)


def build_auto_llm(options: AgentRuntimeOptions):
    providers = []
    if get_gemini_api_key() and not env_flag("COMPARAG_DISABLE_GEMINI"):
        providers.append(("gemini", GeminiChatLLM(model=options.gemini_model)))
    if get_openai_api_key():
        providers.append(("openai", OpenAIChatLLM()))
    if find_codex_command():
        providers.append(("codex_testing", CodexTestingLLM()))
    providers.append(("retrieval_fallback", ExtractiveFallbackLLM()))
    return ProviderFallbackLLM(providers)


def resolve_llm_mode(mode: str) -> str:
    if mode != "auto":
        return mode
    if codex_testing_enabled():
        return "codex_testing"
    return "auto"


def context_profile_for_options(options: AgentRuntimeOptions):
    mode = resolve_llm_mode(options.llm)
    if mode == "codex_testing":
        return resolve_context_profile("codex_testing")
    if mode in {"gemini", "auto"}:
        return resolve_context_profile("gemini", options.gemini_model)
    return resolve_context_profile("small")


def build_memory(options: AgentRuntimeOptions):
    if options.memory_backend == "local":
        return InMemoryConversationMemory()
    if options.memory_backend == "supabase":
        return SupabaseConversationMemory()
    if options.memory_backend == "auto" and supabase_available():
        return FallbackConversationMemory(SupabaseConversationMemory())
    return InMemoryConversationMemory()


def build_reranker(options: AgentRuntimeOptions):
    if not options.enable_reranker:
        return None
    return LocalCrossEncoderReranker(
        model_name=resolve_reranker_model(options.reranker_model),
        device=options.reranker_device,
        allow_download=options.allow_reranker_download,
    )


def resolve_chat_embedding_model(embedding_model: str | None, record: dict[str, Any]) -> str:
    if embedding_model:
        return resolve_embedding_model(embedding_model)
    if isinstance(record.get("embedding_model"), str) and record["embedding_model"]:
        return str(record["embedding_model"])
    return DEFAULT_EMBEDDING_MODEL


def resolve_chat_collection_name(collection: str | None, record: dict[str, Any], embedding_model: str) -> str:
    if collection:
        return collection
    if record.get("embedding_model") == embedding_model and isinstance(record.get("collection_name"), str):
        return str(record["collection_name"])
    return collection_name_for_embedding(DEFAULT_COLLECTION, embedding_model)


def engine_cache_key(
    comparison_id: str,
    options: AgentRuntimeOptions,
    app_dir: Path,
    embedding_model: str,
    collection_name: str,
) -> tuple[Any, ...]:
    return (
        comparison_id,
        str(app_dir),
        str(resolve_chroma_dir(options.chroma_dir)),
        embedding_model,
        collection_name,
        resolve_llm_mode(options.llm),
        options.gemini_model,
        options.retrieval_mode,
        options.semantic_weight,
        options.lexical_weight,
        options.enable_reranker,
        options.reranker_model,
        options.reranker_device,
        options.memory_backend,
    )
