from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..storage import DEFAULT_APP_DIR
from ..vector_store import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION, collection_name_for_embedding


def resolve_app_dir(value: str | None) -> Path:
    return Path(value) if value else DEFAULT_APP_DIR


def resolve_chroma_dir(value: str | None) -> Path:
    return Path(value) if value else DEFAULT_CHROMA_DIR


def resolve_collection_name(collection: str | None, embedding_model: str) -> str:
    if collection:
        return collection
    return collection_name_for_embedding(DEFAULT_COLLECTION, embedding_model)


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "comparison_id": record.get("comparison_id"),
        "chunk_count": record.get("chunk_count"),
        "embedding_model": record.get("embedding_model"),
        "collection_name": record.get("collection_name"),
        "observability_run_id": record.get("observability_run_id"),
        "videos": record.get("videos") or [],
    }
