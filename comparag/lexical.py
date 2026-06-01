from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import RagChunk, RetrievedChunk
from .storage import DEFAULT_APP_DIR, safe_filename


LEXICAL_DIR = "lexical"
TOKEN_PATTERN = re.compile(r"[#@]?\w+", flags=re.UNICODE)
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "ask",
    "based",
    "between",
    "both",
    "by",
    "can",
    "compare",
    "did",
    "do",
    "does",
    "each",
    "for",
    "from",
    "get",
    "give",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "me",
    "mention",
    "mentions",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "video",
    "what",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class LexicalDocument:
    id: str
    text: str
    metadata: dict[str, Any]
    tokens: list[str]


class BM25LexicalIndex:
    def __init__(
        self,
        documents: list[LexicalDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(document.tokens) for document in documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_frequencies = [Counter(document.tokens) for document in documents]
        self.document_frequencies = build_document_frequencies(documents)

    @classmethod
    def from_chunks(cls, chunks: list[RagChunk]) -> "BM25LexicalIndex":
        documents = [
            LexicalDocument(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.chroma_metadata(),
                tokens=tokenize(chunk.text),
            )
            for chunk in chunks
            if chunk.text.strip()
        ]
        return cls(documents)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "documents": [
                {
                    "id": document.id,
                    "text": document.text,
                    "metadata": document.metadata,
                    "tokens": document.tokens,
                }
                for document in self.documents
            ],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "BM25LexicalIndex":
        documents = [
            LexicalDocument(
                id=str(item["id"]),
                text=str(item["text"]),
                metadata=dict(item.get("metadata") or {}),
                tokens=[str(token) for token in item.get("tokens") or tokenize(item.get("text") or "")],
            )
            for item in payload.get("documents") or []
        ]
        return cls(documents)

    def query(
        self,
        query_text: str,
        *,
        comparison_id: str,
        video_id: str | None = None,
        doc_types: list[str] | None = None,
        n_results: int = 6,
    ) -> list[RetrievedChunk]:
        query_terms = tokenize_query(query_text)
        if not query_terms:
            return []

        candidates: list[tuple[float, LexicalDocument]] = []
        for index, document in enumerate(self.documents):
            if not document_matches(document, comparison_id=comparison_id, video_id=video_id, doc_types=doc_types):
                continue
            score = self.score_document(query_terms, index)
            if score > 0:
                candidates.append((score, document))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(
                id=document.id,
                text=document.text,
                metadata={
                    **document.metadata,
                    "retrieval_source": "lexical",
                    "lexical_score": round(score, 6),
                },
                distance=None,
            )
            for score, document in candidates[:n_results]
        ]

    def score_document(self, query_terms: list[str], document_index: int) -> float:
        score = 0.0
        term_frequency = self.term_frequencies[document_index]
        doc_length = self.doc_lengths[document_index]
        for term in query_terms:
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue
            idf = self.idf(term)
            denominator = frequency + self.k1 * (1 - self.b + self.b * doc_length / (self.avg_doc_length or 1.0))
            score += idf * ((frequency * (self.k1 + 1)) / denominator)
        return score

    def idf(self, term: str) -> float:
        document_count = len(self.documents)
        document_frequency = self.document_frequencies.get(term, 0)
        return math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))


class LazyBM25LexicalIndex:
    def __init__(self, *, comparison_id: str, app_dir: Path = DEFAULT_APP_DIR):
        self.comparison_id = comparison_id
        self.app_dir = app_dir
        self._index: BM25LexicalIndex | None = None

    @property
    def index(self) -> BM25LexicalIndex:
        if self._index is None:
            self._index = load_lexical_index(self.comparison_id, self.app_dir)
        return self._index

    def query(
        self,
        query_text: str,
        *,
        comparison_id: str,
        video_id: str | None = None,
        doc_types: list[str] | None = None,
        n_results: int = 6,
    ) -> list[RetrievedChunk]:
        return self.index.query(
            query_text,
            comparison_id=comparison_id,
            video_id=video_id,
            doc_types=doc_types,
            n_results=n_results,
        )


def save_lexical_index(
    chunks: list[RagChunk],
    *,
    comparison_id: str,
    app_dir: Path = DEFAULT_APP_DIR,
) -> Path:
    index = BM25LexicalIndex.from_chunks(chunks)
    path = lexical_index_path(app_dir, comparison_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"comparison_id": comparison_id, **index.to_json()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_lexical_index(comparison_id: str, app_dir: Path = DEFAULT_APP_DIR) -> BM25LexicalIndex:
    path = lexical_index_path(app_dir, comparison_id)
    if not path.exists():
        raise FileNotFoundError(f"No lexical index found at {path}. Run `python -m comparag index` first.")
    return BM25LexicalIndex.from_json(json.loads(path.read_text(encoding="utf-8")))


def lexical_index_path(app_dir: Path, comparison_id: str) -> Path:
    return app_dir / LEXICAL_DIR / f"{safe_filename(comparison_id)}.json"


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def tokenize_query(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in QUERY_STOPWORDS]


def build_document_frequencies(documents: list[LexicalDocument]) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for document in documents:
        frequencies.update(set(document.tokens))
    return frequencies


def document_matches(
    document: LexicalDocument,
    *,
    comparison_id: str,
    video_id: str | None,
    doc_types: list[str] | None,
) -> bool:
    if document.metadata.get("comparison_id") != comparison_id:
        return False
    if video_id and document.metadata.get("video_id") != video_id:
        return False
    if doc_types and document.metadata.get("doc_type") not in set(doc_types):
        return False
    return True
