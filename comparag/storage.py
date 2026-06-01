from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import VideoProfile


DEFAULT_APP_DIR = Path(".cache") / "comparag"
COMPARISONS_DIR = "comparisons"


def comparison_path(app_dir: Path, comparison_id: str) -> Path:
    return app_dir / COMPARISONS_DIR / f"{safe_filename(comparison_id)}.json"


def save_comparison_record(
    *,
    comparison_id: str,
    profiles: list[VideoProfile],
    chunk_count: int,
    source_path: str | None,
    embedding_model: str | None = None,
    collection_name: str | None = None,
    analysis_artifacts: dict[str, Any] | None = None,
    comment_facts: dict[str, Any] | None = None,
    chunk_manifest: dict[str, Any] | None = None,
    indexing_stats: dict[str, Any] | None = None,
    observability_run_id: str | None = None,
    app_dir: Path = DEFAULT_APP_DIR,
) -> Path:
    path = comparison_path(app_dir, comparison_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "comparison_id": comparison_id,
        "source_path": source_path,
        "chunk_count": chunk_count,
        "embedding_model": embedding_model,
        "collection_name": collection_name,
        "observability_run_id": observability_run_id,
        "videos": [profile.to_dict() for profile in profiles],
    }
    if chunk_manifest:
        record["chunk_manifest"] = chunk_manifest
    if comment_facts:
        record["comment_facts"] = comment_facts
    if indexing_stats:
        record["indexing_stats"] = indexing_stats
    if analysis_artifacts:
        record.update(analysis_artifacts)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_comparison_record(comparison_id: str, app_dir: Path = DEFAULT_APP_DIR) -> dict[str, Any]:
    path = comparison_path(app_dir, comparison_id)
    if not path.exists():
        raise FileNotFoundError(f"No comparison record found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def safe_filename(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        else:
            keep.append("_")
    filename = "".join(keep).strip("._")
    return filename or "comparison"
