from __future__ import annotations

import os
from pathlib import Path


GEMINI_API_KEY_ENV_NAMES = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "GEMINI_KEY",
)
OPENAI_API_KEY_ENV_NAMES = ("OPENAI_API_KEY", "OPENAI_KEY")
CODEX_TESTING_MODE = "codex_testing"


def load_dotenv_file(path: str | Path | None = None) -> None:
    env_path = Path(path) if path else Path.cwd() / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_gemini_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for env_name in GEMINI_API_KEY_ENV_NAMES:
        value = os.environ.get(env_name)
        if value:
            return value
    load_dotenv_file()
    for env_name in GEMINI_API_KEY_ENV_NAMES:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def gemini_key_available() -> bool:
    return bool(get_gemini_api_key())


def get_openai_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for env_name in OPENAI_API_KEY_ENV_NAMES:
        value = os.environ.get(env_name)
        if value:
            return value
    load_dotenv_file()
    for env_name in OPENAI_API_KEY_ENV_NAMES:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def openai_key_available() -> bool:
    return bool(get_openai_api_key())


def get_llm_mode() -> str:
    load_dotenv_file()
    return (
        os.environ.get("COMPARAG_LLM_MODE")
        or os.environ.get("LLM_MODE")
        or os.environ.get("MODE")
        or ""
    ).strip().lower()


def codex_testing_enabled() -> bool:
    if get_llm_mode() == CODEX_TESTING_MODE:
        return True
    value = (os.environ.get("CODEX_TESTING") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_supabase_url() -> str | None:
    load_dotenv_file()
    return os.environ.get("SUPABASE_URL")


def get_supabase_service_role_key() -> str | None:
    load_dotenv_file()
    return os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")


def supabase_available() -> bool:
    return bool(get_supabase_url() and get_supabase_service_role_key())


def env_flag(name: str) -> bool:
    value = (os.environ.get(name) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}
