from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Any

from .storage import DEFAULT_APP_DIR, safe_filename


OBSERVABILITY_DIR = "observability"
SECRET_KEY_PARTS = ("api_key", "authorization", "cookie", "password", "secret", "session", "token")


def new_run_id(prefix: str = "run") -> str:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"


class ObservabilityLogger:
    def __init__(self, *, run_id: str | None = None, app_dir: Path = DEFAULT_APP_DIR):
        self.run_id = run_id or new_run_id()
        self.path = app_dir / OBSERVABILITY_DIR / f"{safe_filename(self.run_id)}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, name: str, payload: dict[str, Any] | None = None) -> None:
        record = {
            "ts": dt.datetime.now(dt.UTC).isoformat(),
            "run_id": self.run_id,
            "event": name,
            "payload": scrub_secrets(payload or {}),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def scrub_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        scrubbed = {}
        for key, nested in value.items():
            if any(part in str(key).lower() for part in SECRET_KEY_PARTS):
                scrubbed[key] = "[REDACTED]"
            else:
                scrubbed[key] = scrub_secrets(nested)
        return scrubbed
    if isinstance(value, list):
        return [scrub_secrets(item) for item in value]
    return value
