from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int = 0


class InMemoryRateLimiter:
    def __init__(self):
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, *, limit: int, window_seconds: int) -> RateLimitDecision:
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                retry_after = max(1, int(window_seconds - (now - events[0])) + 1)
                return RateLimitDecision(False, limit, 0, retry_after)
            events.append(now)
            return RateLimitDecision(True, limit, max(0, limit - len(events)), 0)


class DuplicateRequestCache:
    def __init__(self, *, ttl_seconds: float = 20.0, wait_timeout_seconds: float = 300.0):
        self.ttl_seconds = ttl_seconds
        self.wait_timeout_seconds = wait_timeout_seconds
        self._entries: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def run(self, key: str, fn: Callable[[], JsonDict]) -> tuple[JsonDict, bool]:
        now = time.monotonic()
        with self._condition:
            self._prune_locked(now)
            entry = self._entries.get(key)
            if entry and entry["status"] == "done":
                return dict(entry["result"]), True
            if entry and entry["status"] == "running":
                deadline = now + self.wait_timeout_seconds
                while entry["status"] == "running":
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Duplicate request is still running.")
                    self._condition.wait(timeout=remaining)
                if entry["status"] == "done":
                    return dict(entry["result"]), True
                if entry.get("error"):
                    raise RuntimeError(str(entry["error"]))
            self._entries[key] = {"status": "running", "created_at": now}

        try:
            result = fn()
        except Exception as exc:
            with self._condition:
                self._entries[key] = {
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "created_at": now,
                    "expires_at": time.monotonic() + self.ttl_seconds,
                }
                self._condition.notify_all()
            raise

        with self._condition:
            self._entries[key] = {
                "status": "done",
                "result": dict(result),
                "created_at": now,
                "expires_at": time.monotonic() + self.ttl_seconds,
            }
            self._condition.notify_all()
        return result, False

    def _prune_locked(self, now: float) -> None:
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.get("status") != "running" and float(entry.get("expires_at") or 0) <= now
        ]
        for key in expired:
            self._entries.pop(key, None)


class StreamDuplicateGuard:
    def __init__(self, *, ttl_seconds: float = 20.0):
        self.ttl_seconds = ttl_seconds
        self._active: set[str] = set()
        self._completed: dict[str, tuple[float, list[JsonDict]]] = {}
        self._lock = threading.Lock()

    def stream(self, key: str, fn: Callable[[], Iterable[JsonDict]]) -> Iterable[JsonDict]:
        with self._lock:
            self._prune_locked(time.monotonic())
            cached = self._completed.get(key)
            if cached:
                for event in cached[1]:
                    yield dict(event)
                return
            if key in self._active:
                yield {
                    "type": "duplicate",
                    "duplicate": True,
                    "message": "Duplicate stream request is already in progress.",
                }
                yield {"type": "done", "duplicate": True}
                return
            self._active.add(key)

        events: list[JsonDict] = []
        try:
            for event in fn():
                event_dict = dict(event)
                events.append(event_dict)
                yield event_dict
        finally:
            with self._lock:
                self._active.discard(key)
                self._completed[key] = (time.monotonic() + self.ttl_seconds, events)

    def _prune_locked(self, now: float) -> None:
        expired = [key for key, (expires_at, _) in self._completed.items() if expires_at <= now]
        for key in expired:
            self._completed.pop(key, None)


def fingerprint_payload(*, namespace: str, payload: Any, explicit_key: str | None = None) -> str:
    if explicit_key:
        source = f"{namespace}:explicit:{explicit_key}"
    else:
        source = f"{namespace}:payload:{stable_json(payload)}"
    return "idem_" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]


def stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
