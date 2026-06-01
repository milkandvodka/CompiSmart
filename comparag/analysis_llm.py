from __future__ import annotations

import json
import time
from typing import Any

import requests

from .config import get_gemini_api_key


DEFAULT_ANALYSIS_MODEL = "gemini-2.5-flash-lite"


def gemini_json(
    prompt: str,
    *,
    api_key: str | None = None,
    model: str = DEFAULT_ANALYSIS_MODEL,
    timeout_seconds: float = 60,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> dict[str, Any]:
    resolved_key = get_gemini_api_key(api_key)
    if not resolved_key:
        raise RuntimeError("A Gemini API key is required for LLM analysis.")
    response = post_json_with_retries(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        api_key=resolved_key,
        prompt=prompt,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    text_response = payload["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text_response)


def post_json_with_retries(
    url: str,
    *,
    api_key: str,
    prompt: str,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
):
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }
    last_response = None
    for attempt in range(max_retries + 1):
        response = requests.post(
            url,
            headers={"x-goog-api-key": api_key},
            json=payload,
            timeout=timeout_seconds,
        )
        if response.status_code not in {429, 500, 502, 503, 504} or attempt >= max_retries:
            return response
        last_response = response
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.isdigit() else retry_backoff_seconds * (2 ** attempt)
        time.sleep(delay)
    return last_response
