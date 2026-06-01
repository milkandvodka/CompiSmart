from __future__ import annotations

from typing import Any, Iterable

from ..models import JsonDict, RagChunk, VideoProfile
from .utils import (
    as_float,
    chunk_id,
    clean_text,
    dedupe_chunks,
    format_time_range,
    seconds_label,
    stable_hash,
    truncate,
)

DEFAULT_WINDOW_SECONDS = 12
DEFAULT_OVERLAP_SECONDS = 2


def build_transcript_chunks(video: JsonDict, profile: VideoProfile) -> list[RagChunk]:
    transcript = video.get("transcript") or {}
    text = transcript_text(transcript, preferred="english_normalized")
    if not text:
        return []

    raw_text = transcript_text(transcript, preferred="original") or text
    hinglish_text = transcript_text(transcript, preferred="hinglish_latin") or text
    duration = profile.duration_seconds
    segments = normalized_segments(transcript.get("segments") or [])
    chunks: list[RagChunk] = []

    chunks.append(
        make_transcript_chunk(
            profile,
            doc_type="full_transcript",
            text=text,
            display_text=raw_text,
            raw_text=raw_text,
            hinglish_text=hinglish_text,
            start_seconds=None,
            end_seconds=duration,
            is_time_approximate=False,
        )
    )

    for end_seconds, doc_type in [(5, "hook_0_5s"), (10, "hook_0_10s")]:
        hook_text, raw_hook_text, approximate = timed_text_window(
            text=text,
            raw_text=raw_text,
            segments=segments,
            start_seconds=0,
            end_seconds=end_seconds,
            duration_seconds=duration,
        )
        if hook_text:
            chunks.append(
                make_transcript_chunk(
                    profile,
                    doc_type=doc_type,
                    text=hook_text,
                    display_text=raw_hook_text or hook_text,
                    raw_text=raw_hook_text or hook_text,
                    hinglish_text=slice_by_time_ratio(hinglish_text, 0, end_seconds, duration),
                    start_seconds=0,
                    end_seconds=end_seconds,
                    is_time_approximate=approximate,
                )
            )

    if has_usable_timestamps(segments):
        chunks.extend(build_timed_windows(profile, text, raw_text, hinglish_text, segments, duration))
    else:
        chunks.extend(build_word_windows(profile, text, raw_text, hinglish_text))

    return dedupe_chunks(chunks)


def build_timed_windows(
    profile: VideoProfile,
    text: str,
    raw_text: str,
    hinglish_text: str,
    segments: list[dict[str, Any]],
    duration_seconds: int | float | None,
) -> list[RagChunk]:
    if not duration_seconds:
        return build_word_windows(profile, text, raw_text, hinglish_text)

    chunks: list[RagChunk] = []
    start = 0
    window = DEFAULT_WINDOW_SECONDS
    step = max(1, window - DEFAULT_OVERLAP_SECONDS)
    while start < duration_seconds:
        end = min(float(duration_seconds), start + window)
        window_text, raw_window_text, approximate = timed_text_window(
            text=text,
            raw_text=raw_text,
            segments=segments,
            start_seconds=start,
            end_seconds=end,
            duration_seconds=duration_seconds,
        )
        if window_text:
            chunks.append(
                make_transcript_chunk(
                    profile,
                    doc_type="transcript_window",
                    text=window_text,
                    display_text=raw_window_text or window_text,
                    raw_text=raw_window_text or window_text,
                    hinglish_text=slice_by_time_ratio(hinglish_text, start, end, duration_seconds),
                    start_seconds=start,
                    end_seconds=end,
                    is_time_approximate=approximate,
                )
            )
        if end >= duration_seconds:
            break
        start += step
    return chunks


def build_word_windows(
    profile: VideoProfile,
    text: str,
    raw_text: str,
    hinglish_text: str,
    *,
    window_words: int = 90,
    overlap_words: int = 20,
) -> list[RagChunk]:
    words = text.split()
    if not words:
        return []
    chunks: list[RagChunk] = []
    step = max(1, window_words - overlap_words)
    for start in range(0, len(words), step):
        end = min(len(words), start + window_words)
        window_text = " ".join(words[start:end])
        raw_window = word_slice(raw_text, start, end)
        hinglish_window = word_slice(hinglish_text, start, end)
        chunks.append(
            make_transcript_chunk(
                profile,
                doc_type="transcript_text_window",
                text=window_text,
                display_text=raw_window or window_text,
                raw_text=raw_window or window_text,
                hinglish_text=hinglish_window or window_text,
                start_seconds=None,
                end_seconds=None,
                is_time_approximate=True,
                ordinal=len(chunks),
            )
        )
        if end >= len(words):
            break
    return chunks


def make_transcript_chunk(
    profile: VideoProfile,
    *,
    doc_type: str,
    text: str,
    display_text: str,
    raw_text: str,
    hinglish_text: str,
    start_seconds: int | float | None,
    end_seconds: int | float | None,
    is_time_approximate: bool,
    ordinal: int | None = None,
) -> RagChunk:
    time_label = format_time_range(start_seconds, end_seconds)
    citation_label = f"Video {profile.video_id}, transcript {time_label}" if time_label else f"Video {profile.video_id}, transcript"
    if is_time_approximate and time_label:
        citation_label += " approx"
    chunk_key = f"{doc_type}:{time_label or ordinal or 'full'}:{stable_hash(text)}"
    return RagChunk(
        id=chunk_id(profile, "transcript", chunk_key),
        comparison_id=profile.comparison_id,
        video_id=profile.video_id,
        doc_type=doc_type,
        text=f"Video {profile.video_id} {doc_type.replace('_', ' ')}.\n{text}",
        display_text=display_text,
        metadata={
            "platform": profile.platform,
            "source_url": profile.url,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "is_time_approximate": is_time_approximate,
            "raw_text": truncate(raw_text, 2000),
            "hinglish_text": truncate(hinglish_text, 2000),
            "citation_label": citation_label,
        },
    )


def transcript_text(transcript: JsonDict, *, preferred: str) -> str:
    variants = transcript.get("variants") or {}
    if preferred == "english_normalized":
        return clean_text(variants.get("english_normalized") or transcript.get("text") or "")
    if preferred == "hinglish_latin":
        return clean_text(variants.get("hinglish_latin") or transcript.get("text") or "")
    if preferred == "original":
        return clean_text(variants.get("original") or transcript.get("text") or "")
    return clean_text(transcript.get("text") or "")


def normalized_segments(segments: Iterable[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        start = as_float(segment.get("start"))
        duration = as_float(segment.get("duration"))
        text = clean_text(segment.get("text") or "")
        if start is None or not text:
            continue
        end = start + duration if duration is not None else None
        normalized.append({"start": start, "end": end, "text": text})
    return normalized


def timed_text_window(
    *,
    text: str,
    raw_text: str,
    segments: list[dict[str, Any]],
    start_seconds: int | float,
    end_seconds: int | float,
    duration_seconds: int | float | None,
) -> tuple[str, str, bool]:
    if has_usable_timestamps(segments):
        raw_window = " ".join(
            segment["text"]
            for segment in segments
            if segment.get("end") is not None and segment["start"] < end_seconds and segment["end"] > start_seconds
        )
        normalized_window = slice_by_time_ratio(text, start_seconds, end_seconds, duration_seconds)
        return clean_text(normalized_window or raw_window), clean_text(raw_window), False
    if duration_seconds:
        return (
            slice_by_time_ratio(text, start_seconds, end_seconds, duration_seconds),
            slice_by_time_ratio(raw_text, start_seconds, end_seconds, duration_seconds),
            True,
        )
    return text, raw_text, True


def has_usable_timestamps(segments: list[dict[str, Any]]) -> bool:
    if len(segments) < 2:
        return False
    return all(segment.get("end") is not None for segment in segments)


def slice_by_time_ratio(text: str, start_seconds: int | float, end_seconds: int | float, duration_seconds: int | float | None) -> str:
    if not text or not duration_seconds:
        return text
    words = text.split()
    if not words:
        return ""
    start_index = int((float(start_seconds) / float(duration_seconds)) * len(words))
    end_index = max(start_index + 1, int((float(end_seconds) / float(duration_seconds)) * len(words)))
    return " ".join(words[start_index:min(len(words), end_index)])


def word_slice(text: str, start: int, end: int) -> str:
    words = text.split()
    if not words:
        return ""
    return " ".join(words[start:min(len(words), end)])
