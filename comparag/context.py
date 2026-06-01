from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelContextProfile:
    name: str
    context_window_tokens: int
    max_prompt_chars: int
    max_history_chars: int
    max_memory_summary_chars: int
    max_retrieved_context_chars: int
    max_retrieved_chunk_chars: int
    max_tool_result_chars: int
    memory_summary_trigger_messages: int = 12
    memory_recent_messages_for_summary: int = 12


DEFAULT_CONTEXT_PROFILE = ModelContextProfile(
    name="default",
    context_window_tokens=32000,
    max_prompt_chars=36000,
    max_history_chars=3000,
    max_memory_summary_chars=5000,
    max_retrieved_context_chars=9000,
    max_retrieved_chunk_chars=1800,
    max_tool_result_chars=4000,
)


CONTEXT_PROFILES = {
    "gemini": ModelContextProfile(
        name="gemini",
        context_window_tokens=1000000,
        max_prompt_chars=60000,
        max_history_chars=5000,
        max_memory_summary_chars=7000,
        max_retrieved_context_chars=16000,
        max_retrieved_chunk_chars=2400,
        max_tool_result_chars=6000,
    ),
    "codex_testing": ModelContextProfile(
        name="codex_testing",
        context_window_tokens=128000,
        max_prompt_chars=50000,
        max_history_chars=4000,
        max_memory_summary_chars=6000,
        max_retrieved_context_chars=12000,
        max_retrieved_chunk_chars=2200,
        max_tool_result_chars=6000,
    ),
    "openai": ModelContextProfile(
        name="openai",
        context_window_tokens=128000,
        max_prompt_chars=50000,
        max_history_chars=4000,
        max_memory_summary_chars=6000,
        max_retrieved_context_chars=12000,
        max_retrieved_chunk_chars=2200,
        max_tool_result_chars=6000,
    ),
    "small": ModelContextProfile(
        name="small",
        context_window_tokens=16000,
        max_prompt_chars=20000,
        max_history_chars=1800,
        max_memory_summary_chars=2500,
        max_retrieved_context_chars=6000,
        max_retrieved_chunk_chars=1200,
        max_tool_result_chars=2500,
    ),
}


def resolve_context_profile(provider: str | None = None, model: str | None = None) -> ModelContextProfile:
    key = (provider or "").strip().lower()
    model_key = (model or "").strip().lower()
    if key in CONTEXT_PROFILES:
        return CONTEXT_PROFILES[key]
    if "gemini" in model_key:
        return CONTEXT_PROFILES["gemini"]
    if model_key.startswith(("gpt-", "o1", "o3", "o4")):
        return CONTEXT_PROFILES["openai"]
    if "mini" in model_key or "small" in model_key:
        return CONTEXT_PROFILES["small"]
    return DEFAULT_CONTEXT_PROFILE
