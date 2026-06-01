from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable, Protocol

import requests

from .config import get_gemini_api_key, get_openai_api_key


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_CODEX_TESTING_MODEL = ""
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_TIMEOUT_SECONDS = 60.0
DEFAULT_GEMINI_MAX_RETRIES = 3
DEFAULT_OPENAI_TIMEOUT_SECONDS = 60.0
DEFAULT_OPENAI_MAX_RETRIES = 1


class ChatLLM(Protocol):
    def complete(self, prompt: str) -> str:
        ...

    def stream(self, prompt: str) -> Iterable[str]:
        ...


class GeminiChatLLM:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        retry_backoff_seconds: float = 2.0,
    ):
        self.api_key = get_gemini_api_key(api_key)
        if not self.api_key:
            raise RuntimeError("A Gemini API key is required for GeminiChatLLM.")
        self.model = model
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else env_float(
            "GEMINI_TIMEOUT_SECONDS",
            DEFAULT_GEMINI_TIMEOUT_SECONDS,
        )
        self.max_retries = max_retries if max_retries is not None else env_int(
            "GEMINI_MAX_RETRIES",
            DEFAULT_GEMINI_MAX_RETRIES,
        )
        self.retry_backoff_seconds = retry_backoff_seconds

    def complete(self, prompt: str) -> str:
        return "".join(self.stream(prompt))

    def stream(self, prompt: str) -> Iterable[str]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent"
        response = self.post_stream_with_retries(url, prompt)
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = json.loads(line.removeprefix("data: ").strip())
            for candidate in payload.get("candidates", []):
                content = candidate.get("content") or {}
                for part in content.get("parts") or []:
                    text = part.get("text")
                    if text:
                        yield text

    def post_stream_with_retries(self, url: str, prompt: str):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        last_response = None
        for attempt in range(self.max_retries + 1):
            response = requests.post(
                url,
                headers={"x-goog-api-key": self.api_key},
                params={"alt": "sse"},
                json=payload,
                timeout=self.timeout_seconds,
                stream=True,
            )
            if response.status_code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                return response
            last_response = response
            response.close()
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else self.retry_backoff_seconds * (2 ** attempt)
            time.sleep(delay)
        return last_response


class OpenAIChatLLM:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        retry_backoff_seconds: float = 1.0,
    ):
        self.api_key = get_openai_api_key(api_key)
        if not self.api_key:
            raise RuntimeError("An OpenAI API key is required for OpenAIChatLLM.")
        self.model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else env_float(
            "OPENAI_TIMEOUT_SECONDS",
            DEFAULT_OPENAI_TIMEOUT_SECONDS,
        )
        self.max_retries = max_retries if max_retries is not None else env_int(
            "OPENAI_MAX_RETRIES",
            DEFAULT_OPENAI_MAX_RETRIES,
        )
        self.retry_backoff_seconds = retry_backoff_seconds

    def complete(self, prompt: str) -> str:
        return "".join(self.stream(prompt))

    def stream(self, prompt: str) -> Iterable[str]:
        response = self.post_stream_with_retries(prompt)
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                break
            payload = json.loads(data)
            for choice in payload.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    yield text

    def post_stream_with_retries(self, prompt: str):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "stream": True,
        }
        last_response = None
        for attempt in range(self.max_retries + 1):
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout_seconds,
                stream=True,
            )
            if response.status_code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= self.max_retries:
                return response
            last_response = response
            response.close()
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else self.retry_backoff_seconds * (2 ** attempt)
            time.sleep(delay)
        return last_response


class ProviderFallbackLLM:
    def __init__(self, providers: list[tuple[str, ChatLLM]]):
        self.providers = providers

    def complete(self, prompt: str) -> str:
        return "".join(self.stream(prompt))

    def stream(self, prompt: str) -> Iterable[str]:
        errors = []
        for name, provider in self.providers:
            try:
                emitted = False
                for token in provider.stream(prompt):
                    emitted = True
                    yield token
                return
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                if emitted:
                    raise RuntimeError(f"{name} failed after streaming began: {type(exc).__name__}: {exc}") from exc
        raise RuntimeError("All LLM providers failed. " + " | ".join(errors))


class ExtractiveFallbackLLM:
    """No-network fallback that exposes retrieved context clearly."""

    def complete(self, prompt: str) -> str:
        return (
            "I do not have an LLM API key configured, so this is a retrieval-only draft.\n\n"
            + prompt[-3500:]
        )

    def stream(self, prompt: str) -> Iterable[str]:
        answer = self.complete(prompt)
        for token in answer.split(" "):
            yield token + " "


class CodexTestingLLM:
    """Test-only provider that shells out to `codex exec` instead of Gemini."""

    def __init__(
        self,
        *,
        command: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 120,
        cwd: str | Path | None = None,
    ):
        self.command = command or os.environ.get("CODEX_CLI_PATH") or find_codex_command()
        self.model = model or os.environ.get("CODEX_TESTING_MODEL") or DEFAULT_CODEX_TESTING_MODEL
        self.timeout_seconds = timeout_seconds
        self.cwd = Path(cwd) if cwd else None

    def complete(self, prompt: str) -> str:
        if not self.command:
            raise RuntimeError("codex_testing mode requested, but no codex CLI command was found.")
        with tempfile.TemporaryDirectory(prefix="comparag_codex_") as temp_dir:
            run_dir = self.cwd or Path(temp_dir) / "empty_workspace"
            run_dir.mkdir(parents=True, exist_ok=True)
            output_path = Path(temp_dir) / "last_message.txt"
            codex_prompt = build_codex_testing_prompt(prompt)
            command = [
                self.command,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-rules",
                "--sandbox",
                "read-only",
                "--cd",
                str(run_dir),
                "--output-last-message",
                str(output_path),
                "-",
            ]
            if self.model:
                command[2:2] = ["--model", self.model]
            completed = subprocess.run(
                command,
                input=codex_prompt,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=self.timeout_seconds,
                cwd=str(run_dir),
            )
            if completed.returncode != 0:
                stderr = completed.stderr.strip()[-1000:]
                raise RuntimeError(f"codex_testing provider failed with exit code {completed.returncode}: {stderr}")
            if output_path.exists():
                answer = output_path.read_text(encoding="utf-8").strip()
                if answer:
                    return answer
            stdout = completed.stdout.strip()
            if stdout:
                return stdout
            raise RuntimeError("codex_testing provider returned an empty response.")

    def stream(self, prompt: str) -> Iterable[str]:
        answer = self.complete(prompt)
        for token in answer.split(" "):
            yield token + " "


def find_codex_command() -> str | None:
    for name in ("codex.cmd", "codex"):
        found = shutil.which(name)
        if found:
            return found
    return None


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, ""))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, ""))
    except ValueError:
        return default


def build_codex_testing_prompt(rag_prompt: str) -> str:
    return f"""
You are running as a test-only LLM provider for a local RAG pipeline.

Rules:
- Do not run shell commands, inspect files, browse, or use tools.
- Answer only from the supplied RAG prompt below.
- Preserve the RAG prompt's citation rules exactly.
- Use only citation labels listed in "Available citation labels".
- If the RAG prompt lacks evidence for a requested fact, say it is unavailable.
- Keep the answer concise and useful.

RAG prompt:
{rag_prompt}
""".strip()
