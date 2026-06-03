from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable

from .analysis_llm import DEFAULT_ANALYSIS_MODEL
from .comment_intelligence import analyze_comments
from .config import gemini_key_available
from .creative_features import analyze_creative_features
from .models import JsonDict, VideoProfile
from .observability import ObservabilityLogger


ProgressCallback = Callable[[str, str, float | None, dict[str, Any] | None], None]


@dataclass(frozen=True)
class AnalysisConfig:
    comment_intelligence: str = "llm"
    creative_features: str = "llm"
    analysis_model: str = DEFAULT_ANALYSIS_MODEL
    max_comment_evidence_chars: int = 12000
    force_analysis_refresh: bool = False


def enrich_payload_with_analysis(
    payload: JsonDict,
    *,
    profiles: list[VideoProfile],
    previous_record: JsonDict | None,
    config: AnalysisConfig,
    logger: ObservabilityLogger | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[JsonDict, JsonDict]:
    enriched = copy.deepcopy(payload)
    videos = enriched.get("videos") or []
    previous_comments = (previous_record or {}).get("comment_intelligence") or {}
    previous_creative = (previous_record or {}).get("creative_features") or {}
    comment_results: dict[str, JsonDict] = {}
    creative_results: dict[str, JsonDict] = {}
    fingerprints = {"comment_intelligence": {}, "creative_features": {}}

    total_videos = max(1, len(videos))
    for video_index, (profile, video) in enumerate(zip(profiles, videos)):
        analysis = video.setdefault("analysis", {})
        comments = [comment for comment in video.get("public_comment_objects") or [] if isinstance(comment, dict)]
        emit_analysis_progress(
            progress,
            video_index=video_index,
            total_videos=total_videos,
            substep=0,
            stage="analysis_comment_intelligence",
            message=f"Analyzing Video {profile.video_id} comments.",
            details={"video_id": profile.video_id, "comment_count": len(comments), "mode": config.comment_intelligence},
        )
        comment_start = time.perf_counter()
        comment_result = analyze_comments(
            comments,
            video_id=profile.video_id,
            creator_username=creator_username_candidate(profile),
            mode=config.comment_intelligence,
            model=config.analysis_model,
            max_evidence_chars=config.max_comment_evidence_chars,
            cached=previous_comments.get(profile.video_id) if isinstance(previous_comments, dict) else None,
            force_refresh=config.force_analysis_refresh,
        )
        log_stage(
            logger,
            "comment_intelligence",
            profile.video_id,
            comment_start,
            {
                "mode": config.comment_intelligence,
                "llm_used": comment_result.get("llm_used"),
                "cache_hit": comment_result.get("cache_hit"),
                "comments": len(comments),
                "clusters": comment_result.get("cluster_count"),
                "compression_ratio": comment_result.get("compression_ratio"),
                "evidence_pack_chars": comment_result.get("evidence_pack_chars"),
                "warnings": comment_result.get("warnings"),
            },
        )
        emit_analysis_progress(
            progress,
            video_index=video_index,
            total_videos=total_videos,
            substep=1,
            stage="analysis_comment_intelligence_done",
            message=f"Video {profile.video_id} comment intelligence ready.",
            details={
                "video_id": profile.video_id,
                "llm_used": comment_result.get("llm_used"),
                "cache_hit": comment_result.get("cache_hit"),
                "clusters": comment_result.get("cluster_count"),
                "evidence_pack_chars": comment_result.get("evidence_pack_chars"),
            },
        )

        emit_analysis_progress(
            progress,
            video_index=video_index,
            total_videos=total_videos,
            substep=2,
            stage="analysis_creative_features",
            message=f"Analyzing Video {profile.video_id} transcript creative features.",
            details={"video_id": profile.video_id, "mode": config.creative_features},
        )
        creative_start = time.perf_counter()
        creative_result = analyze_creative_features(
            video,
            video_id=profile.video_id,
            mode=config.creative_features,
            model=config.analysis_model,
            cached=previous_creative.get(profile.video_id) if isinstance(previous_creative, dict) else None,
            force_refresh=config.force_analysis_refresh,
        )
        log_stage(
            logger,
            "creative_features",
            profile.video_id,
            creative_start,
            {
                "mode": config.creative_features,
                "llm_used": creative_result.get("llm_used"),
                "cache_hit": creative_result.get("cache_hit"),
                "available": creative_result.get("available"),
                "warnings": creative_result.get("warnings"),
            },
        )
        emit_analysis_progress(
            progress,
            video_index=video_index,
            total_videos=total_videos,
            substep=3,
            stage="analysis_creative_features_done",
            message=f"Video {profile.video_id} creative features ready.",
            details={
                "video_id": profile.video_id,
                "llm_used": creative_result.get("llm_used"),
                "cache_hit": creative_result.get("cache_hit"),
                "available": creative_result.get("available"),
            },
        )

        analysis["comment_intelligence"] = comment_result
        analysis["creative_features"] = creative_result
        comment_results[profile.video_id] = comment_result
        creative_results[profile.video_id] = creative_result
        fingerprints["comment_intelligence"][profile.video_id] = comment_result.get("fingerprint")
        fingerprints["creative_features"][profile.video_id] = creative_result.get("fingerprint")

    artifacts = {
        "comment_intelligence": comment_results,
        "creative_features": creative_results,
        "analysis_fingerprints": fingerprints,
        "analysis_model": config.analysis_model,
        "analysis_modes": {
            "comment_intelligence": config.comment_intelligence,
            "creative_features": config.creative_features,
        },
        "analysis_warnings": analysis_warnings(comment_results, creative_results),
        "gemini_key_available": gemini_key_available(),
    }
    return enriched, artifacts


def creator_username_candidate(profile: VideoProfile) -> str | None:
    for value in (profile.creator_id, profile.creator):
        if not value:
            continue
        candidate = str(value).strip().lstrip("@")
        if candidate and " " not in candidate:
            return candidate
    return None


def analysis_warnings(comment_results: dict[str, JsonDict], creative_results: dict[str, JsonDict]) -> list[str]:
    warnings = []
    for results in (comment_results, creative_results):
        for video_id, result in results.items():
            for warning in result.get("warnings") or []:
                warnings.append(f"Video {video_id}: {warning}")
    return warnings


def log_stage(
    logger: ObservabilityLogger | None,
    stage: str,
    video_id: str,
    start: float,
    payload: JsonDict,
) -> None:
    if not logger:
        return
    logger.event(
        stage,
        {
            "video_id": video_id,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            **payload,
        },
    )


def emit_analysis_progress(
    progress: ProgressCallback | None,
    *,
    video_index: int,
    total_videos: int,
    substep: int,
    stage: str,
    message: str,
    details: JsonDict,
) -> None:
    if not progress:
        return
    total_steps = max(1, total_videos * 4)
    completed_steps = min(total_steps, (video_index * 4) + substep)
    percent = 18.0 + (17.0 * (completed_steps / total_steps))
    progress(stage, message, percent, details)
