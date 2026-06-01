from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from ..analysis import AnalysisConfig, enrich_payload_with_analysis
from ..chunking import build_chunks
from ..comment_facts import build_comment_fact_table
from ..embeddings import resolve_embedding_model
from ..indexing import plan_index_update
from ..lexical import save_lexical_index
from ..metrics import build_video_profiles
from ..observability import ObservabilityLogger
from ..storage import comparison_path, load_comparison_record, save_comparison_record
from ..vector_store import ChromaChunkStore
from .paths import load_json_file, resolve_app_dir, resolve_chroma_dir, resolve_collection_name
from .schemas import ExtractAndIndexRequest, IndexOptions


ProgressCallback = Callable[[str, str, float | None, dict[str, Any] | None], None]


def index_payload(
    payload: dict[str, Any],
    *,
    comparison_id: str,
    source_path: str | None,
    options: IndexOptions,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    app_dir = resolve_app_dir(options.app_dir)
    logger = ObservabilityLogger(app_dir=app_dir)
    logger.event("api_index_start", {"comparison_id": comparison_id, "source_path": source_path})
    emit_progress(progress, "profiles", "Building structured video profiles and metrics.", 8.0)
    profiles = build_video_profiles(payload, comparison_id)
    previous_record = load_previous_record(comparison_id, app_dir)
    emit_progress(progress, "analysis", "Preparing comment intelligence and creative evidence.", 18.0)
    enriched_payload, analysis_artifacts = enrich_payload_with_analysis(
        payload,
        profiles=profiles,
        previous_record=previous_record,
        config=AnalysisConfig(
            comment_intelligence=options.comment_intelligence,
            creative_features=options.creative_features,
            analysis_model=options.analysis_model,
            max_comment_evidence_chars=options.max_comment_evidence_chars,
            force_analysis_refresh=options.force_analysis_refresh,
        ),
        logger=logger,
    )
    emit_progress(progress, "chunking", "Chunking transcripts, comments, metrics, and creative evidence.", 35.0)
    profiles, chunks = build_chunks(enriched_payload, comparison_id)
    comment_facts = build_comment_fact_table(enriched_payload, profiles)
    embedding_model = resolve_embedding_model(options.embedding_model)
    collection_name = resolve_collection_name(options.collection, embedding_model)
    emit_progress(
        progress,
        "planning_index",
        "Checking incremental index manifest for changed chunks.",
        45.0,
        {"chunk_count": len(chunks), "embedding_model": embedding_model},
    )
    index_plan = plan_index_update(
        chunks,
        previous_record=previous_record,
        force_reindex=options.force_reindex,
        no_delete=options.no_reset,
    )
    vector_write_needed = (
        bool(index_plan.chunks_to_upsert)
        or bool(index_plan.chunk_ids_to_delete)
        or (index_plan.mode in {"force_reindex", "full_no_previous_manifest"} and not options.no_reset)
    )
    if vector_write_needed:
        emit_progress(
            progress,
            "embedding",
            "Loading embedding model and embedding changed chunks.",
            60.0,
            {"upsert_count": index_plan.upsert_count, "collection_name": collection_name},
        )
        store = ChromaChunkStore(
            persist_dir=resolve_chroma_dir(options.chroma_dir),
            collection_name=collection_name,
            embedding_model=embedding_model,
            embedding_device=options.embedding_device,
            allow_embedding_download=options.allow_embedding_download,
        )
        if index_plan.mode in {"force_reindex", "full_no_previous_manifest"} and not options.no_reset:
            store.reset_comparison(comparison_id)
        elif index_plan.chunk_ids_to_delete:
            store.delete_chunks(index_plan.chunk_ids_to_delete)
        store.upsert_chunks(index_plan.chunks_to_upsert)
    else:
        emit_progress(progress, "embedding", "Skipping embeddings because chunks are unchanged.", 75.0)
    emit_progress(progress, "lexical_index", "Writing BM25 lexical index.", 88.0)
    lexical_path = save_lexical_index(chunks, comparison_id=comparison_id, app_dir=app_dir)
    emit_progress(progress, "saving_record", "Saving comparison record and observability metadata.", 96.0)
    record_path = save_comparison_record(
        comparison_id=comparison_id,
        profiles=profiles,
        chunk_count=len(chunks),
        source_path=source_path,
        app_dir=app_dir,
        embedding_model=embedding_model,
        collection_name=collection_name,
        analysis_artifacts=analysis_artifacts,
        comment_facts=comment_facts,
        chunk_manifest=index_plan.chunk_manifest,
        indexing_stats=index_plan.stats(),
        observability_run_id=logger.run_id,
    )
    result = {
        "comparison_id": comparison_id,
        "chunk_count": len(chunks),
        "record_path": str(record_path),
        "lexical_path": str(lexical_path),
        "chroma_dir": str(resolve_chroma_dir(options.chroma_dir)),
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "observability_run_id": logger.run_id,
        "indexing": index_plan.stats(),
        "vector_write_needed": vector_write_needed,
        "comment_fact_count": len(comment_facts.get("facts") or []),
    }
    logger.event("api_index_complete", result)
    emit_progress(progress, "indexed", "Indexing complete.", 99.0, {"chunk_count": len(chunks)})
    return result


def build_extractor_command(request: ExtractAndIndexRequest, output_path: Path) -> list[str]:
    extraction = request.extraction
    command = [
        sys.executable,
        "social_video_extractor.py",
        "--youtube-url",
        request.youtube_url,
        "--instagram-url",
        request.instagram_url,
        "--language",
        extraction.language,
        "--output",
        str(output_path),
        "--asr-provider",
        extraction.asr_provider,
        "--asr-model",
        extraction.asr_model,
        "--hf-asr-model",
        extraction.hf_asr_model,
        "--asr-timeout-seconds",
        str(extraction.asr_timeout_seconds),
        "--asr-device",
        extraction.asr_device,
        "--asr-compute-type",
        extraction.asr_compute_type,
    ]
    if extraction.fetch_comments:
        command.extend(["--fetch-comments", "--max-comments", str(extraction.max_comments)])
    if extraction.comment_time_budget_seconds is not None:
        command.extend(["--comment-time-budget-seconds", str(extraction.comment_time_budget_seconds)])
    if extraction.fetch_comment_replies:
        command.extend(["--fetch-comment-replies", "--max-comment-replies", str(extraction.max_comment_replies)])
    if extraction.instagrapi_settings:
        command.extend(["--instagrapi-settings", extraction.instagrapi_settings])
    if extraction.cookies:
        command.extend(["--cookies", extraction.cookies])
    if extraction.cookies_from_browser:
        command.extend(["--cookies-from-browser", extraction.cookies_from_browser])
    if extraction.asr_language:
        command.extend(["--asr-language", extraction.asr_language])
    if extraction.no_asr:
        command.append("--no-asr")
    if extraction.require_transcripts:
        command.append("--require-transcripts")
    return command


def load_previous_record(comparison_id: str, app_dir: Path) -> dict[str, Any] | None:
    path = comparison_path(app_dir, comparison_id)
    if not path.exists():
        return None
    return load_comparison_record(comparison_id, app_dir)


def safe_job_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value).strip("._") or "comparison"


def emit_progress(
    progress: ProgressCallback | None,
    stage: str,
    message: str,
    percent: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    if progress:
        progress(stage, message, percent, details)


def scale_progress(progress: ProgressCallback | None, *, start: float, end: float) -> ProgressCallback | None:
    if progress is None:
        return None

    def scaled(stage: str, message: str, percent: float | None = None, details: dict[str, Any] | None = None) -> None:
        mapped = None
        if percent is not None:
            mapped = start + ((end - start) * (float(percent) / 100.0))
        progress(stage, message, mapped, details)

    return scaled
