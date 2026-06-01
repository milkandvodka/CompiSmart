from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .models import RagChunk


CHUNK_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IndexUpdatePlan:
    mode: str
    chunk_manifest: dict[str, Any]
    chunks_to_upsert: list[RagChunk]
    chunk_ids_to_delete: list[str]
    unchanged_count: int

    @property
    def upsert_count(self) -> int:
        return len(self.chunks_to_upsert)

    @property
    def delete_count(self) -> int:
        return len(self.chunk_ids_to_delete)

    @property
    def total_count(self) -> int:
        return len(self.chunk_manifest.get("chunks") or {})

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "total_chunks": self.total_count,
            "upsert_count": self.upsert_count,
            "delete_count": self.delete_count,
            "unchanged_count": self.unchanged_count,
        }


def plan_index_update(
    chunks: list[RagChunk],
    *,
    previous_record: dict[str, Any] | None,
    force_reindex: bool = False,
    no_delete: bool = False,
) -> IndexUpdatePlan:
    manifest = build_chunk_manifest(chunks)
    if force_reindex:
        return IndexUpdatePlan(
            mode="force_reindex",
            chunk_manifest=manifest,
            chunks_to_upsert=chunks,
            chunk_ids_to_delete=[],
            unchanged_count=0,
        )

    previous_chunks = previous_chunk_manifest(previous_record)
    if not previous_chunks:
        return IndexUpdatePlan(
            mode="full_no_previous_manifest",
            chunk_manifest=manifest,
            chunks_to_upsert=chunks,
            chunk_ids_to_delete=[],
            unchanged_count=0,
        )

    current_chunks = manifest["chunks"]
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    changed_or_new_ids = [
        chunk_id
        for chunk_id, info in current_chunks.items()
        if previous_chunks.get(chunk_id, {}).get("fingerprint") != info.get("fingerprint")
    ]
    deleted_ids = [] if no_delete else sorted(set(previous_chunks) - set(current_chunks))
    unchanged_count = len(current_chunks) - len(changed_or_new_ids)
    return IndexUpdatePlan(
        mode="incremental",
        chunk_manifest=manifest,
        chunks_to_upsert=[chunks_by_id[chunk_id] for chunk_id in changed_or_new_ids],
        chunk_ids_to_delete=deleted_ids,
        unchanged_count=unchanged_count,
    )


def build_chunk_manifest(chunks: list[RagChunk]) -> dict[str, Any]:
    return {
        "schema_version": CHUNK_MANIFEST_SCHEMA_VERSION,
        "chunks": {
            chunk.id: {
                "fingerprint": chunk_fingerprint(chunk),
                "video_id": chunk.video_id,
                "doc_type": chunk.doc_type,
                "citation_label": chunk.citation_label(),
            }
            for chunk in chunks
        },
    }


def previous_chunk_manifest(previous_record: dict[str, Any] | None) -> dict[str, Any]:
    manifest = (previous_record or {}).get("chunk_manifest") or {}
    if manifest.get("schema_version") != CHUNK_MANIFEST_SCHEMA_VERSION:
        return {}
    chunks = manifest.get("chunks")
    return chunks if isinstance(chunks, dict) else {}


def chunk_fingerprint(chunk: RagChunk) -> str:
    payload = {
        "text": chunk.text,
        "metadata": chunk.chroma_metadata(),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
