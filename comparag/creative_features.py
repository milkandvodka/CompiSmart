from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .analysis_llm import DEFAULT_ANALYSIS_MODEL, gemini_json


JsonDict = dict[str, Any]
CREATIVE_ARRAY_FIELDS = {
    "pain_points",
    "proof_elements",
    "claims",
    "risk_flags",
    "improvement_opportunities",
    "notes",
}
CREATIVE_TEXT_FIELDS = {
    "hook_type",
    "first_5s_promise",
    "target_audience",
    "cta",
    "emotional_angle",
}


def transcript_fingerprint(transcript: JsonDict) -> str:
    payload = {
        "text": transcript.get("text"),
        "variants": transcript.get("variants"),
        "segments": transcript.get("segments"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def analyze_creative_features(
    video: JsonDict,
    *,
    video_id: str,
    mode: str,
    model: str = DEFAULT_ANALYSIS_MODEL,
    cached: JsonDict | None = None,
    force_refresh: bool = False,
) -> JsonDict:
    transcript = video.get("transcript") or {}
    fingerprint = transcript_fingerprint(transcript)
    if cached and not force_refresh and cached.get("fingerprint") == fingerprint:
        return {**cached, "cache_hit": True}

    if mode == "off":
        return {
            **empty_creative_features(),
            "mode": "off",
            "fingerprint": fingerprint,
            "cache_hit": False,
            "note": "Creative feature extraction disabled.",
        }

    evidence = prepare_creative_evidence(video, video_id=video_id)
    evidence.update(
        {
            "fingerprint": fingerprint,
            "cache_hit": False,
            "llm_used": False,
            "model": None,
            "mode": mode,
        }
    )

    if mode != "llm":
        evidence["warnings"].append("Creative feature inference requires LLM mode; only transcript evidence was prepared.")
        return evidence

    if not evidence["evidence_available"]:
        evidence["warnings"].append("No transcript text available for creative feature extraction.")
        return evidence

    try:
        llm_result = coerce_creative_result(llm_creative_features(evidence["evidence_pack"], model=model))
    except Exception as exc:
        evidence["warnings"].append(f"LLM creative feature extraction failed; no semantic fallback was generated: {exc}")
        return evidence

    merged = {**evidence, **llm_result}
    merged["available"] = True
    merged["analysis_available"] = True
    merged["llm"] = llm_result
    merged["llm_used"] = True
    merged["model"] = model
    return merged


def prepare_creative_evidence(video: JsonDict, *, video_id: str) -> JsonDict:
    transcript = video.get("transcript") or {}
    text = transcript_variant(transcript, "english_normalized")
    hook_5s = first_seconds_text(transcript, fallback_text=text, seconds=5)
    hook_10s = first_seconds_text(transcript, fallback_text=text, seconds=10)
    evidence_pack = build_creative_evidence_pack(
        video_id=video_id,
        text=text,
        hook_5s=hook_5s,
        hook_10s=hook_10s,
    )
    return {
        **empty_creative_features(),
        "evidence_available": bool(text),
        "hook_text": hook_5s,
        "first_5s_transcript": hook_5s,
        "first_10s_transcript": hook_10s,
        "transcript_chars": len(text),
        "evidence_pack": evidence_pack,
        "evidence_pack_chars": len(evidence_pack),
        "warnings": [],
    }


def empty_creative_features() -> JsonDict:
    return {
        "available": False,
        "analysis_available": False,
        "evidence_available": False,
        "hook_type": None,
        "first_5s_promise": None,
        "target_audience": None,
        "pain_points": [],
        "proof_elements": [],
        "cta": None,
        "emotional_angle": None,
        "claims": [],
        "risk_flags": [],
        "improvement_opportunities": [],
        "notes": [],
        "llm_used": False,
        "model": None,
        "warnings": [],
    }


def llm_creative_features(evidence_pack: str, *, model: str) -> JsonDict:
    prompt = f"""
Extract transcript-only creative features for a short-form social video. Return strict JSON only with:
- hook_type: string or null
- first_5s_promise: string or null
- target_audience: string or null
- pain_points: array
- proof_elements: array
- cta: string or null
- emotional_angle: string or null
- claims: array
- risk_flags: array
- improvement_opportunities: array
- notes: array

Rules:
- Use only the transcript evidence below.
- Do not infer visual, speaker-tone, audio, product, creator, or audience details that are not in the transcript.
- If a field is not supported by the transcript, return null or an empty array for that field.
- Keep claims faithful to the wording in the transcript; put uncertainty in notes.
- Do not use keyword matching shortcuts. Judge the meaning of the transcript.

Evidence:
{evidence_pack}
""".strip()
    return gemini_json(prompt, model=model)


def coerce_creative_result(value: JsonDict) -> JsonDict:
    result: JsonDict = {}
    source = value if isinstance(value, dict) else {}
    for field in CREATIVE_TEXT_FIELDS:
        raw = source.get(field)
        result[field] = clean_text(raw) if raw not in (None, "") else None
    for field in CREATIVE_ARRAY_FIELDS:
        raw_items = source.get(field)
        if isinstance(raw_items, list):
            result[field] = [clean_text(item) for item in raw_items if clean_text(item)]
        elif raw_items in (None, ""):
            result[field] = []
        else:
            result[field] = [clean_text(raw_items)]
    return result


def build_creative_evidence_pack(*, video_id: str, text: str, hook_5s: str, hook_10s: str) -> str:
    return "\n".join(
        [
            f"Video {video_id} transcript-only creative evidence.",
            f"First 5 seconds transcript: {hook_5s or '[unavailable]'}",
            f"First 10 seconds transcript: {hook_10s or '[unavailable]'}",
            f"Full normalized transcript: {fit_to_budget(text, 16000) or '[unavailable]'}",
        ]
    )


def transcript_variant(transcript: JsonDict, variant: str) -> str:
    variants = transcript.get("variants") or {}
    return clean_text(variants.get(variant) or transcript.get("text") or "")


def first_seconds_text(transcript: JsonDict, *, fallback_text: str, seconds: int) -> str:
    segments = [segment for segment in transcript.get("segments") or [] if isinstance(segment, dict)]
    pieces = []
    for segment in segments:
        start = as_float(segment.get("start"))
        if start is None:
            continue
        if start < seconds:
            pieces.append(str(segment.get("text") or ""))
    if pieces and len(segments) > 1:
        return clean_text(" ".join(pieces))
    words = fallback_text.split()
    if not words:
        return ""
    estimated_words = max(1, min(len(words), int(len(words) * 0.18)))
    return " ".join(words[:estimated_words])


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def fit_to_budget(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "..."


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
