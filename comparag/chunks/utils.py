from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from ..models import JsonDict, RagChunk, VideoProfile


def format_time_range(start: int | float | None, end: int | float | None) -> str | None:
    if start is None and end is None:
        return None
    if start is None:
        return f"00:00-{seconds_label(end)}"
    if end is None:
        return f"{seconds_label(start)}+"
    return f"{seconds_label(start)}-{seconds_label(end)}"


def seconds_label(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    total = int(round(float(value)))
    minutes = total // 60
    seconds = total % 60
    return f"{minutes:02d}:{seconds:02d}"


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "..."


def grouped(items: list[JsonDict], size: int) -> Iterable[list[JsonDict]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def dedupe_chunks(chunks: list[RagChunk]) -> list[RagChunk]:
    seen: set[tuple[str, str]] = set()
    deduped: list[RagChunk] = []
    for chunk in chunks:
        key = (chunk.doc_type, chunk.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def chunk_id(profile: VideoProfile, *parts: str) -> str:
    clean_parts = [safe_id_part(profile.comparison_id), profile.video_id]
    clean_parts.extend(safe_id_part(part) for part in parts if part)
    return "_".join(clean_parts)


def safe_id_part(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_")[:96]


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
