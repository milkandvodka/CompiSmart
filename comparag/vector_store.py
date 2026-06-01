from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .embeddings import DEFAULT_EMBEDDING_MODEL, LocalSentenceTransformerEmbedding
from .models import RagChunk, RetrievedChunk


DEFAULT_CHROMA_DIR = Path(".cache") / "chroma"
DEFAULT_COLLECTION = "comparag_chunks"


class ChromaChunkStore:
    def __init__(
        self,
        *,
        persist_dir: Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_function: Any | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_device: str | None = None,
        allow_embedding_download: bool = False,
    ):
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Missing dependency: chromadb. Install with: pip install -r requirements.txt") from exc

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_function = embedding_function or LocalSentenceTransformerEmbedding(
            embedding_model,
            device=embedding_device,
            allow_download=allow_embedding_download,
        )
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_comparison(self, comparison_id: str) -> None:
        try:
            self.collection.delete(where={"comparison_id": comparison_id})
        except Exception:
            # Chroma raises when nothing matches in some versions. Empty reset is fine.
            return

    def delete_chunks(self, chunk_ids: list[str], *, batch_size: int = 128) -> None:
        for start in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[start:start + batch_size]
            if batch:
                self.collection.delete(ids=batch)

    def upsert_chunks(self, chunks: list[RagChunk], *, batch_size: int = 128) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            if not batch:
                continue
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk.chroma_metadata() for chunk in batch],
            )

    def query(
        self,
        query_text: str,
        *,
        comparison_id: str,
        video_id: str | None = None,
        doc_types: list[str] | None = None,
        n_results: int = 6,
    ) -> list[RetrievedChunk]:
        where = build_where_filter(comparison_id=comparison_id, video_id=video_id, doc_types=doc_types)
        result = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return unpack_query_result(result)


class LazyChromaChunkStore:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self._store: ChromaChunkStore | None = None

    @property
    def store(self) -> ChromaChunkStore:
        if self._store is None:
            self._store = ChromaChunkStore(**self.kwargs)
        return self._store

    def query(
        self,
        query_text: str,
        *,
        comparison_id: str,
        video_id: str | None = None,
        doc_types: list[str] | None = None,
        n_results: int = 6,
    ) -> list[RetrievedChunk]:
        return self.store.query(
            query_text,
            comparison_id=comparison_id,
            video_id=video_id,
            doc_types=doc_types,
            n_results=n_results,
        )


def build_where_filter(
    *,
    comparison_id: str,
    video_id: str | None = None,
    doc_types: list[str] | None = None,
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = [{"comparison_id": {"$eq": comparison_id}}]
    if video_id:
        filters.append({"video_id": {"$eq": video_id}})
    if doc_types:
        if len(doc_types) == 1:
            filters.append({"doc_type": {"$eq": doc_types[0]}})
        else:
            filters.append({"doc_type": {"$in": doc_types}})
    return filters[0] if len(filters) == 1 else {"$and": filters}


def unpack_query_result(result: dict[str, Any]) -> list[RetrievedChunk]:
    ids = first_result_list(result.get("ids"))
    documents = first_result_list(result.get("documents"))
    metadatas = first_result_list(result.get("metadatas"))
    distances = first_result_list(result.get("distances"))

    chunks: list[RetrievedChunk] = []
    for index, chunk_id in enumerate(ids):
        chunks.append(
            RetrievedChunk(
                id=str(chunk_id),
                text=str(documents[index]) if index < len(documents) else "",
                metadata=dict(metadatas[index] or {}) if index < len(metadatas) else {},
                distance=float(distances[index]) if index < len(distances) and distances[index] is not None else None,
            )
        )
    return chunks


def first_result_list(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []


def collection_name_for_embedding(base_collection: str, embedding_model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", embedding_model).strip("._-")
    name = f"{base_collection}_{slug}"[:120].strip("._-")
    return name if len(name) >= 3 else base_collection
