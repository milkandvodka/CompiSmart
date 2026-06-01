from __future__ import annotations

import os
from typing import Iterable


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_MODEL_PRESETS = {
    "fast": "sentence-transformers/all-MiniLM-L6-v2",
    "balanced": "intfloat/multilingual-e5-base",
    "quality": "BAAI/bge-m3",
    "bge-m3": "BAAI/bge-m3",
    "e5-base": "intfloat/multilingual-e5-base",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


class LocalSentenceTransformerEmbedding:
    """Small local transformer embedding adapter compatible with Chroma."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str | None = None,
        allow_download: bool = False,
    ):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
            from transformers.utils import logging as transformers_logging  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: transformers/torch. Install with: pip install -r requirements.txt"
            ) from exc

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
        self.model_name = model_name
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        local_files_only = not allow_download
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        except Exception as exc:
            if not allow_download:
                raise RuntimeError(
                    f"Embedding model {model_name!r} is not fully cached locally. "
                    "Run the index command once with --allow-embedding-download."
                ) from exc
            raise
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, input: list[str]) -> list[list[float]]:  # Chroma expects this exact argument name.
        return self.embed_documents(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._embed(add_embedding_prefixes(input, self.model_name, kind="query"))

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self._embed(add_embedding_prefixes(input, self.model_name, kind="document"))

    def _embed(self, input: list[str]) -> list[list[float]]:
        encoded = self.tokenizer(
            list(input),
            padding=True,
            truncation=True,
            max_length=384,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            output = self.model(**encoded)
        token_embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        embeddings = summed / counts
        embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

    def name(self) -> str:
        return f"sentence-transformers:{self.model_name}"


class HashTestEmbedding:
    """Tiny hash embedding for unit tests; not for real retrieval quality."""

    def __init__(self, dimensions: int = 16):
        self.dimensions = dimensions

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in input]

    def embed_one(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for index, byte in enumerate(text.encode("utf-8")):
            values[index % self.dimensions] += byte / 255.0
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / norm for value in values]

    def name(self) -> str:
        return "hash-test-embedding"


def embedding_texts(chunks: Iterable[object]) -> list[str]:
    return [str(getattr(chunk, "text")) for chunk in chunks]


def resolve_embedding_model(model_or_preset: str) -> str:
    return EMBEDDING_MODEL_PRESETS.get(model_or_preset, model_or_preset)


def add_embedding_prefixes(texts: list[str], model_name: str, *, kind: str) -> list[str]:
    lowered = model_name.lower()
    if "e5" not in lowered:
        return texts
    prefix = "query: " if kind == "query" else "passage: "
    return [text if text.lower().startswith(prefix) else f"{prefix}{text}" for text in texts]
