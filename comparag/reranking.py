from __future__ import annotations

import os
from typing import Any

from .models import RetrievedChunk


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_MODEL_PRESETS = {
    "quality": "BAAI/bge-reranker-base",
    "bge-base": "BAAI/bge-reranker-base",
    "fast": "cross-encoder/ms-marco-MiniLM-L6-v2",
}


class LocalCrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str | None = None,
        allow_download: bool = False,
        batch_size: int = 8,
        max_length: int = 512,
    ):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            import torch  # type: ignore
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
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
        self.batch_size = batch_size
        self.max_length = max_length
        local_files_only = not allow_download
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            if not allow_download:
                raise RuntimeError(
                    f"Reranker model {model_name!r} is not fully cached locally. "
                    "Run chat with --allow-reranker-download once, or disable reranking."
                ) from exc
            raise
        self.model.to(self.device)
        self.model.eval()

    def rerank(self, query: str, chunks: list[RetrievedChunk], *, top_n: int) -> list[RetrievedChunk]:
        if not chunks:
            return []

        scored: list[tuple[float, RetrievedChunk]] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start:start + self.batch_size]
            encoded = self.tokenizer(
                [query] * len(batch),
                [chunk.text for chunk in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.no_grad():
                output = self.model(**encoded)
            scores = logits_to_scores(output.logits)
            scored.extend((float(score), chunk) for score, chunk in zip(scores, batch))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(
                id=chunk.id,
                text=chunk.text,
                metadata={**chunk.metadata, "rerank_score": round(score, 6), "reranker_model": self.model_name},
                distance=chunk.distance,
            )
            for score, chunk in scored[:top_n]
        ]


class NoopReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk], *, top_n: int) -> list[RetrievedChunk]:
        return chunks[:top_n]


def logits_to_scores(logits: Any) -> list[float]:
    if len(logits.shape) == 1:
        return logits.detach().cpu().tolist()
    if logits.shape[-1] == 1:
        return logits.squeeze(-1).detach().cpu().tolist()
    return logits[:, -1].detach().cpu().tolist()


def resolve_reranker_model(model_or_preset: str) -> str:
    return RERANKER_MODEL_PRESETS.get(model_or_preset, model_or_preset)
