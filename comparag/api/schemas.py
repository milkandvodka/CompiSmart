from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


LlmMode = Literal["auto", "gemini", "openai", "fallback", "codex_testing"]
RetrievalMode = Literal["semantic", "lexical", "hybrid"]
MemoryBackend = Literal["auto", "local", "supabase"]
AnalysisMode = Literal["off", "evidence", "llm"]


class AgentRuntimeOptions(BaseModel):
    app_dir: str | None = None
    chroma_dir: str | None = None
    collection: str | None = None
    embedding_model: str | None = None
    embedding_device: str | None = None
    allow_embedding_download: bool = False
    llm: LlmMode = "auto"
    gemini_model: str = "gemini-2.5-flash-lite"
    retrieval_mode: RetrievalMode = "hybrid"
    semantic_weight: float = Field(default=1.0, ge=0.0)
    lexical_weight: float = Field(default=1.0, ge=0.0)
    enable_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str | None = None
    allow_reranker_download: bool = False
    memory_backend: MemoryBackend = "auto"


class ChatRequest(BaseModel):
    comparison_id: str
    question: str = Field(min_length=1)
    thread_id: str = "default"
    options: AgentRuntimeOptions = Field(default_factory=AgentRuntimeOptions)
    idempotency_key: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    citation_audit: dict[str, Any]
    evidence_plan: dict[str, Any] | None = None
    route: str | None = None
    llm_error: str | None = None
    memory_summary_update: dict[str, Any] | None = None
    deduped: bool = False


class ComparisonSummary(BaseModel):
    comparison_id: str
    chunk_count: int | None = None
    embedding_model: str | None = None
    collection_name: str | None = None
    observability_run_id: str | None = None
    videos: list[dict[str, Any]]


class ComparisonListResponse(BaseModel):
    comparisons: list[ComparisonSummary]


class IndexOptions(BaseModel):
    app_dir: str | None = None
    chroma_dir: str | None = None
    collection: str | None = None
    embedding_model: str = "balanced"
    embedding_device: str | None = None
    allow_embedding_download: bool = False
    comment_intelligence: AnalysisMode = "llm"
    creative_features: AnalysisMode = "llm"
    analysis_model: str = "gemini-2.5-flash-lite"
    max_comment_evidence_chars: int = Field(default=12000, ge=1000)
    force_analysis_refresh: bool = False
    force_reindex: bool = False
    no_reset: bool = False


class IndexComparisonRequest(BaseModel):
    comparison_id: str
    input_path: str | None = None
    payload: dict[str, Any] | None = None
    options: IndexOptions = Field(default_factory=IndexOptions)
    idempotency_key: str | None = Field(default=None, max_length=128)


class ExtractionOptions(BaseModel):
    language: str = "en"
    fetch_comments: bool = True
    max_comments: int = Field(default=100, ge=0)
    comment_time_budget_seconds: float | None = Field(default=60.0, ge=1.0)
    fetch_comment_replies: bool = False
    max_comment_replies: int = Field(default=0, ge=0)
    instagrapi_settings: str | None = None
    cookies: str | None = None
    cookies_from_browser: str | None = None
    asr_provider: str = "auto"
    asr_model: str = "base"
    hf_asr_model: str = "openai/whisper-large-v3-turbo"
    asr_timeout_seconds: float = Field(default=60.0, ge=5.0)
    asr_language: str | None = None
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    no_asr: bool = False
    require_transcripts: bool = False


class ExtractAndIndexRequest(BaseModel):
    comparison_id: str
    youtube_url: str
    instagram_url: str
    extraction: ExtractionOptions = Field(default_factory=ExtractionOptions)
    index: IndexOptions = Field(default_factory=IndexOptions)
    idempotency_key: str | None = Field(default=None, max_length=128)


class JobResponse(BaseModel):
    job_id: str
    type: str
    status: str
    created_at: str
    updated_at: str
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    progress_events: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None
    deduped: bool = False
