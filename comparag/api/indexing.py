from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

from transcript_normalizer import detect_scripts, normalize_result_payload

from ..analysis import AnalysisConfig, enrich_payload_with_analysis
from ..chunking import build_chunks
from ..comment_facts import build_comment_fact_table
from ..config import codex_testing_enabled, env_flag, get_gemini_api_key, get_openai_api_key
from ..embeddings import resolve_embedding_model
from ..indexing import plan_index_update
from ..lexical import save_lexical_index
from ..llm import CodexTestingLLM, GeminiChatLLM, OpenAIChatLLM, ProviderFallbackLLM, find_codex_command
from ..metrics import build_video_profiles
from ..observability import ObservabilityLogger
from ..storage import comparison_path, load_comparison_record, save_comparison_record
from ..vector_store import ChromaChunkStore
from .paths import load_json_file, resolve_app_dir, resolve_chroma_dir, resolve_collection_name
from .schemas import ExtractAndIndexRequest, IndexOptions


ProgressCallback = Callable[[str, str, float | None, dict[str, Any] | None], None]
TRANSCRIPT_VARIANT_SCHEMA_VERSION = 2


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
    emit_progress(progress, "transcript_normalization", "Preparing transcript variants for retrieval.", 12.0)
    payload, transcript_artifacts = normalize_transcripts_for_indexing(
        payload,
        profiles=profiles,
        previous_record=previous_record,
        options=options,
        logger=logger,
        progress=progress,
    )
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
        progress=progress,
    )
    analysis_artifacts = {**transcript_artifacts, **analysis_artifacts}
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
            "Loading local embedding model.",
            60.0,
            {
                "embedding_model": embedding_model,
                "upsert_count": index_plan.upsert_count,
                "delete_count": index_plan.delete_count,
                "unchanged_count": index_plan.unchanged_count,
                "collection_name": collection_name,
            },
        )
        store = ChromaChunkStore(
            persist_dir=resolve_chroma_dir(options.chroma_dir),
            collection_name=collection_name,
            embedding_model=embedding_model,
            embedding_device=options.embedding_device,
            allow_embedding_download=options.allow_embedding_download,
        )
        emit_progress(
            progress,
            "embedding_model_ready",
            "Embedding model loaded; preparing Chroma writes.",
            68.0,
            {"embedding_model": embedding_model, "collection_name": collection_name},
        )
        if index_plan.mode in {"force_reindex", "full_no_previous_manifest"} and not options.no_reset:
            emit_progress(progress, "vector_reset", "Clearing previous vectors for this comparison.", 70.0)
            store.reset_comparison(comparison_id)
        elif index_plan.chunk_ids_to_delete:
            emit_progress(
                progress,
                "vector_delete",
                "Deleting stale vectors from Chroma.",
                70.0,
                {"delete_count": len(index_plan.chunk_ids_to_delete)},
            )
            store.delete_chunks(index_plan.chunk_ids_to_delete)
        def vector_progress(done: int, total: int) -> None:
            fraction = (done / total) if total else 1.0
            emit_progress(
                progress,
                "vector_upsert",
                f"Embedded and upserted {done}/{total} changed chunks into Chroma.",
                72.0 + (12.0 * fraction),
                {"upserted": done, "upsert_count": total, "collection_name": collection_name},
            )

        store.upsert_chunks(index_plan.chunks_to_upsert, batch_size=16, progress=vector_progress)
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
    default_cookie_path = os.environ.get("INSTAGRAM_COOKIES") or ".cache/instagram-cookies.txt"
    cookies_path = extraction.cookies or (default_cookie_path if Path(default_cookie_path).exists() else None)
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
    if cookies_path:
        command.extend(["--cookies", cookies_path])
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


def normalize_transcripts_for_indexing(
    payload: dict[str, Any],
    *,
    profiles: list[Any],
    previous_record: dict[str, Any] | None,
    options: IndexOptions,
    logger: ObservabilityLogger,
    progress: ProgressCallback | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stats = transcript_normalization_stats(payload)
    if not stats["transcripts_with_text"]:
        emit_progress(progress, "transcript_normalization", "No transcript text available to normalize.", 15.0, stats)
        logger.event("transcript_normalization_skip", stats)
        return payload, {"transcript_variants": {}, "transcript_normalization": stats}

    cache_hits = apply_cached_transcript_variants(payload, profiles, previous_record, force_refresh=options.force_analysis_refresh)
    provider, provider_label = build_transcript_normalization_provider(options.analysis_model)
    try:
        normalized = normalize_result_payload(
            payload,
            api_key=None,
            model=options.analysis_model,
            timeout_seconds=45.0,
            complete=provider.complete if provider else None,
            provider_label=provider_label,
            skip_existing=True,
        )
        result_stats = transcript_normalization_stats(normalized)
        result_stats["provider"] = provider_label or "none"
        result_stats["cache_hits"] = cache_hits
        emit_progress(
            progress,
            "transcript_normalization",
            "Transcript variants ready for embedding and chat context.",
            16.0,
            result_stats,
        )
        logger.event("transcript_normalization_complete", result_stats)
        return normalized, transcript_normalization_artifacts(normalized, profiles, result_stats)
    except Exception as exc:
        fallback = normalize_result_payload(
            payload,
            api_key=None,
            model=options.analysis_model,
            timeout_seconds=0,
            skip_existing=True,
        )
        error_details = {
            **stats,
            "provider": provider_label or "none",
            "cache_hits": cache_hits,
            "error_type": type(exc).__name__,
            "warning": "LLM transcript normalization failed; indexed original transcript text.",
        }
        emit_progress(
            progress,
            "transcript_normalization",
            "Transcript normalization failed; continuing with original transcript text.",
            16.0,
            error_details,
        )
        logger.event("transcript_normalization_error", error_details)
        return fallback, transcript_normalization_artifacts(fallback, profiles, error_details)


def build_transcript_normalization_provider(model: str):
    providers = []
    if get_gemini_api_key() and not env_flag("COMPARAG_DISABLE_GEMINI"):
        providers.append(("gemini", GeminiChatLLM(model=model)))
    if get_openai_api_key():
        providers.append(("openai", OpenAIChatLLM()))
    if codex_testing_enabled() and find_codex_command():
        providers.append(("codex_testing", CodexTestingLLM()))
    if not providers:
        return None, None
    return ProviderFallbackLLM(providers), ",".join(name for name, _provider in providers)


def transcript_normalization_stats(payload: dict[str, Any]) -> dict[str, Any]:
    transcripts_with_text = 0
    llm_used = 0
    detected_scripts: set[str] = set()
    for video in payload.get("videos") or []:
        transcript = video.get("transcript") or {}
        text = str(transcript.get("text") or "")
        if text:
            transcripts_with_text += 1
            detected_scripts.update(detect_scripts(text))
        variants = transcript.get("variants") or {}
        if variants.get("llm_used"):
            llm_used += 1
            detected_scripts.update(str(item) for item in variants.get("detected_scripts") or [] if item)
    return {
        "transcripts_with_text": transcripts_with_text,
        "llm_normalized_transcripts": llm_used,
        "detected_scripts": sorted(detected_scripts),
    }


def apply_cached_transcript_variants(
    payload: dict[str, Any],
    profiles: list[Any],
    previous_record: dict[str, Any] | None,
    *,
    force_refresh: bool,
) -> int:
    if force_refresh:
        return 0
    previous_variants = (previous_record or {}).get("transcript_variants") or {}
    if not isinstance(previous_variants, dict):
        return 0
    cache_hits = 0
    for profile, video in zip(profiles, payload.get("videos") or []):
        transcript = video.get("transcript") or {}
        text = str(transcript.get("text") or "")
        if not text:
            continue
        cached = previous_variants.get(profile.video_id)
        if not isinstance(cached, dict):
            continue
        if cached.get("schema_version") != TRANSCRIPT_VARIANT_SCHEMA_VERSION:
            continue
        if cached.get("fingerprint") != transcript_fingerprint(text):
            continue
        variants = cached.get("variants")
        if not isinstance(variants, dict) or not variants.get("english_normalized"):
            continue
        transcript["variants"] = variants
        video["transcript"] = transcript
        cache_hits += 1
    return cache_hits


def transcript_normalization_artifacts(
    payload: dict[str, Any],
    profiles: list[Any],
    stats: dict[str, Any],
) -> dict[str, Any]:
    variants_by_video: dict[str, Any] = {}
    for profile, video in zip(profiles, payload.get("videos") or []):
        transcript = video.get("transcript") or {}
        text = str(transcript.get("text") or "")
        variants = transcript.get("variants") or {}
        if not text or not isinstance(variants, dict):
            continue
        variants_by_video[profile.video_id] = {
            "schema_version": TRANSCRIPT_VARIANT_SCHEMA_VERSION,
            "fingerprint": transcript_fingerprint(text),
            "variants": variants,
        }
    return {
        "transcript_variants": variants_by_video,
        "transcript_normalization": stats,
    }


def transcript_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
