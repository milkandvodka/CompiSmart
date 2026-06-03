from __future__ import annotations

import json
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterable

from ..rag_graph import RagChatEngine
from ..storage import load_comparison_record
from .indexing import (
    ProgressCallback,
    build_extractor_command,
    emit_progress,
    index_payload,
    load_json_file,
    safe_job_filename,
    scale_progress,
)
from .paths import comparison_summary, resolve_app_dir
from .runtime import build_engine, engine_cache_key, resolve_chat_collection_name, resolve_chat_embedding_model
from .schemas import AgentRuntimeOptions, ChatRequest, ExtractAndIndexRequest, IndexComparisonRequest


def redact_sensitive_text(text: str) -> str:
    redactions = [
        (r"AIza[0-9A-Za-z_-]{20,}", "[redacted-google-key]"),
        (r"sk-[0-9A-Za-z_-]{20,}", "[redacted-openai-key]"),
        (r"hf_[0-9A-Za-z]{20,}", "[redacted-hf-token]"),
        (r"eyJ[0-9A-Za-z_-]{20,}\.[0-9A-Za-z_-]{20,}\.[0-9A-Za-z_-]{20,}", "[redacted-jwt]"),
    ]
    redacted = text
    for pattern, replacement in redactions:
        redacted = re.sub(pattern, replacement, redacted)
    return redacted


def summarize_extractor_stderr(stderr: str) -> str:
    text = redact_sensitive_text((stderr or "").strip())
    if not text:
        return "Extractor exited without an error message."
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.lower().startswith("error:"):
            return line
    for line in reversed(lines):
        if "DownloadError:" in line:
            return line.split("DownloadError:", 1)[-1].strip()
    return text[-1200:]


class AgentService:
    def __init__(self):
        self._engine_cache: dict[tuple[Any, ...], RagChatEngine] = {}
        self._lock = threading.Lock()

    def list_comparisons(self, *, app_dir: str | None = None) -> list[dict[str, Any]]:
        root = resolve_app_dir(app_dir)
        directory = root / "comparisons"
        if not directory.exists():
            return []
        comparisons = []
        for path in sorted(directory.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            comparisons.append(comparison_summary(record))
        return comparisons

    def get_comparison(self, comparison_id: str, *, app_dir: str | None = None) -> dict[str, Any]:
        return load_comparison_record(comparison_id, resolve_app_dir(app_dir))

    def chat(self, request: ChatRequest) -> dict[str, Any]:
        engine = self.engine_for(request.comparison_id, request.options)
        return engine.invoke(
            comparison_id=request.comparison_id,
            question=request.question,
            thread_id=request.thread_id,
        )

    def stream_chat(self, request: ChatRequest) -> Iterable[dict[str, Any]]:
        engine = self.engine_for(request.comparison_id, request.options)
        yield from engine.stream(
            comparison_id=request.comparison_id,
            question=request.question,
            thread_id=request.thread_id,
        )

    def index_comparison(
        self,
        request: IndexComparisonRequest,
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if request.payload is None and not request.input_path:
            raise ValueError("Provide either input_path or payload for indexing.")
        emit_progress(progress, "loading_payload", "Loading extractor payload.", 3.0)
        payload = request.payload if request.payload is not None else load_json_file(Path(str(request.input_path)))
        source_path = request.input_path
        return index_payload(
            payload,
            comparison_id=request.comparison_id,
            source_path=source_path,
            options=request.options,
            progress=progress,
        )

    def extract_and_index(
        self,
        request: ExtractAndIndexRequest,
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        app_dir = resolve_app_dir(request.index.app_dir)
        output_dir = app_dir / "extractions"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{safe_job_filename(request.comparison_id)}.json"
        command = build_extractor_command(request, output_path)
        emit_progress(
            progress,
            "extracting",
            "Running extractor: fetching metadata, comments, captions, and ASR transcript if needed.",
            5.0,
            {
                "comparison_id": request.comparison_id,
                "fetch_comments": request.extraction.fetch_comments,
                "max_comments": request.extraction.max_comments,
                "comment_time_budget_seconds": request.extraction.comment_time_budget_seconds,
                "asr_provider": request.extraction.asr_provider,
                "asr_model": request.extraction.asr_model,
            },
        )
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=max(float(request.extraction.asr_timeout_seconds) + 180.0, 300.0),
        )
        if completed.returncode != 0:
            error_summary = summarize_extractor_stderr(completed.stderr)
            if error_summary.lower().startswith("error:"):
                error_summary = error_summary.split(":", 1)[1].strip()
            raise RuntimeError(f"Extractor failed: {error_summary}")
        emit_progress(
            progress,
            "transcribing",
            "Extraction and transcript generation completed; preparing RAG index.",
            42.0,
            {"output_path": str(output_path)},
        )
        index_result = self.index_comparison(
            IndexComparisonRequest(
                comparison_id=request.comparison_id,
                input_path=str(output_path),
                options=request.index,
            ),
            progress=scale_progress(progress, start=45.0, end=98.0),
        )
        return {
            "extraction_output_path": str(output_path),
            "extractor_stdout_tail": completed.stdout.strip()[-2000:],
            "index": index_result,
        }

    def engine_for(self, comparison_id: str, options: AgentRuntimeOptions) -> RagChatEngine:
        app_dir = resolve_app_dir(options.app_dir)
        record = load_comparison_record(comparison_id, app_dir)
        embedding_model = resolve_chat_embedding_model(options.embedding_model, record)
        collection_name = resolve_chat_collection_name(options.collection, record, embedding_model)
        key = engine_cache_key(comparison_id, options, app_dir, embedding_model, collection_name)
        with self._lock:
            cached = self._engine_cache.get(key)
            if cached is not None:
                return cached
            engine = build_engine(
                comparison_id=comparison_id,
                record=record,
                options=options,
                app_dir=app_dir,
                embedding_model=embedding_model,
                collection_name=collection_name,
            )
            self._engine_cache[key] = engine
            return engine


__all__ = ["AgentService", "comparison_summary"]
