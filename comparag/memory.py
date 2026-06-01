from __future__ import annotations

import datetime as dt
from typing import Any, Protocol

import requests

from .config import get_supabase_service_role_key, get_supabase_url


Message = dict[str, str]
SummaryRecord = dict[str, Any]
COMPARAG_SUPABASE_TABLES = frozenset(
    {
        "comparag_threads",
        "comparag_messages",
        "comparag_memory_summaries",
    }
)


class ConversationMemory(Protocol):
    def get(self, thread_id: str) -> list[Message]:
        ...

    def append_turn(self, thread_id: str, *, user: str, assistant: str) -> None:
        ...

    def get_summary(self, thread_id: str) -> SummaryRecord:
        ...

    def save_summary(self, thread_id: str, summary: str, metadata: dict[str, Any] | None = None) -> None:
        ...

    def message_count(self, thread_id: str) -> int:
        ...


class InMemoryConversationMemory:
    def __init__(self):
        self._threads: dict[str, list[Message]] = {}
        self._summaries: dict[str, SummaryRecord] = {}

    def get(self, thread_id: str) -> list[Message]:
        return list(self._threads.get(thread_id, []))

    def append_turn(self, thread_id: str, *, user: str, assistant: str) -> None:
        messages = self._threads.setdefault(thread_id, [])
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})

    def get_summary(self, thread_id: str) -> SummaryRecord:
        return dict(self._summaries.get(thread_id) or empty_summary_record())

    def save_summary(self, thread_id: str, summary: str, metadata: dict[str, Any] | None = None) -> None:
        self._summaries[thread_id] = {
            "summary": summary,
            "metadata": dict(metadata or {}),
            "updated_at": utc_now(),
        }

    def message_count(self, thread_id: str) -> int:
        return len(self._threads.get(thread_id, []))

    def clear(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)
        self._summaries.pop(thread_id, None)


class FallbackConversationMemory:
    def __init__(self, primary: ConversationMemory, fallback: ConversationMemory | None = None):
        self.primary = primary
        self.fallback = fallback or InMemoryConversationMemory()
        self.last_error: str | None = None

    def get(self, thread_id: str) -> list[Message]:
        try:
            return self.primary.get(thread_id)
        except Exception as exc:
            self.last_error = f"{exc.__class__.__name__}: memory read failed"
            return self.fallback.get(thread_id)

    def append_turn(self, thread_id: str, *, user: str, assistant: str) -> None:
        try:
            self.primary.append_turn(thread_id, user=user, assistant=assistant)
        except Exception as exc:
            self.last_error = f"{exc.__class__.__name__}: memory write failed"
            self.fallback.append_turn(thread_id, user=user, assistant=assistant)

    def get_summary(self, thread_id: str) -> SummaryRecord:
        try:
            return self.primary.get_summary(thread_id)
        except Exception as exc:
            self.last_error = f"{exc.__class__.__name__}: memory summary read failed"
            return self.fallback.get_summary(thread_id)

    def save_summary(self, thread_id: str, summary: str, metadata: dict[str, Any] | None = None) -> None:
        try:
            self.primary.save_summary(thread_id, summary, metadata)
        except Exception as exc:
            self.last_error = f"{exc.__class__.__name__}: memory summary write failed"
            self.fallback.save_summary(thread_id, summary, metadata)

    def message_count(self, thread_id: str) -> int:
        try:
            return self.primary.message_count(thread_id)
        except Exception as exc:
            self.last_error = f"{exc.__class__.__name__}: memory count failed"
            return self.fallback.message_count(thread_id)


class SupabaseConversationMemory:
    def __init__(
        self,
        *,
        url: str | None = None,
        service_role_key: str | None = None,
        max_messages: int = 30,
        timeout_seconds: float = 20,
    ):
        self.url = (url or get_supabase_url() or "").rstrip("/")
        self.service_role_key = service_role_key or get_supabase_service_role_key()
        if not self.url or not self.service_role_key:
            raise RuntimeError("Supabase memory requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
        self.max_messages = max_messages
        self.timeout_seconds = timeout_seconds

    def get(self, thread_id: str) -> list[Message]:
        response = self.session_get(
            "comparag_messages",
            params={
                "select": "role,content,created_at",
                "thread_id": f"eq.{thread_id}",
                "order": "created_at.desc",
                "limit": str(self.max_messages),
            },
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        rows = list(reversed(response.json()))
        return [
            {"role": str(row.get("role") or ""), "content": str(row.get("content") or "")}
            for row in rows
            if row.get("role") and row.get("content")
        ]

    def append_turn(self, thread_id: str, *, user: str, assistant: str) -> None:
        self.upsert_thread(thread_id)
        timestamp = utc_now()
        response = self.session_post(
            "comparag_messages",
            [
                {"thread_id": thread_id, "role": "user", "content": user, "created_at": timestamp},
                {"thread_id": thread_id, "role": "assistant", "content": assistant, "created_at": utc_now()},
            ],
        )
        response.raise_for_status()

    def get_summary(self, thread_id: str) -> SummaryRecord:
        response = self.session_get(
            "comparag_memory_summaries",
            params={
                "select": "summary,metadata,updated_at",
                "thread_id": f"eq.{thread_id}",
                "limit": "1",
            },
        )
        if response.status_code == 404:
            return empty_summary_record()
        response.raise_for_status()
        rows = response.json()
        if not rows:
            return empty_summary_record()
        row = rows[0]
        return {
            "summary": str(row.get("summary") or ""),
            "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
            "updated_at": row.get("updated_at"),
        }

    def save_summary(self, thread_id: str, summary: str, metadata: dict[str, Any] | None = None) -> None:
        self.upsert_thread(thread_id)
        response = self.session_post(
            "comparag_memory_summaries",
            {
                "thread_id": thread_id,
                "summary": summary,
                "metadata": dict(metadata or {}),
                "updated_at": utc_now(),
            },
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
            params={"on_conflict": "thread_id"},
        )
        response.raise_for_status()

    def message_count(self, thread_id: str) -> int:
        response = self.session_get(
            "comparag_messages",
            params={
                "select": "id",
                "thread_id": f"eq.{thread_id}",
                "limit": "1",
            },
            headers={"Prefer": "count=exact"},
        )
        response.raise_for_status()
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            _, total = content_range.rsplit("/", 1)
            try:
                return int(total)
            except ValueError:
                pass
        return len(response.json() or [])

    def upsert_thread(self, thread_id: str) -> None:
        response = self.session_post(
            "comparag_threads",
            {
                "id": thread_id,
                "updated_at": utc_now(),
            },
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
            params={"on_conflict": "id"},
        )
        response.raise_for_status()

    def session_get(self, table: str, *, params: dict[str, str], headers: dict[str, str] | None = None):
        ensure_comparag_table(table)
        merged_headers = self.headers()
        if headers:
            merged_headers.update(headers)
        return requests.get(
            f"{self.url}/rest/v1/{table}",
            headers=merged_headers,
            params=params,
            timeout=self.timeout_seconds,
        )

    def session_post(
        self,
        table: str,
        payload,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ):
        ensure_comparag_table(table)
        merged_headers = {**self.headers(), "Prefer": "return=minimal"}
        if headers:
            merged_headers.update(headers)
        return requests.post(
            f"{self.url}/rest/v1/{table}",
            headers=merged_headers,
            params=params,
            json=payload,
            timeout=self.timeout_seconds,
        )

    def headers(self) -> dict[str, str]:
        return {
            "apikey": self.service_role_key or "",
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": "application/json",
        }


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def empty_summary_record() -> SummaryRecord:
    return {"summary": "", "metadata": {}, "updated_at": None}


def ensure_comparag_table(table: str) -> None:
    if table not in COMPARAG_SUPABASE_TABLES:
        raise ValueError(f"Refusing to access non-compaRAG Supabase table: {table}")
