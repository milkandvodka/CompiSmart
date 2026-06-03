from __future__ import annotations

import datetime as dt
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable


ProgressCallback = Callable[[str, str, float | None, dict[str, Any] | None], None]
JobCallable = Callable[[ProgressCallback], dict[str, Any]]
TERMINAL_STATUSES = {"succeeded", "failed"}


@dataclass
class JobRecord:
    job_id: str
    type: str
    status: str
    created_at: str
    updated_at: str
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    progress_events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None

    def to_dict(self, *, deduped: bool = False) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "type": self.type,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "progress_events": self.progress_events,
            "metadata": self.metadata,
            "idempotency_key": self.idempotency_key,
            "deduped": deduped,
        }


class JobRegistry:
    def __init__(self, *, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="comparag_job")
        self._jobs: dict[str, JobRecord] = {}
        self._idempotency_index: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        job_type: str,
        fn: JobCallable,
        *,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[JobRecord, bool]:
        now = utc_now()
        with self._lock:
            if idempotency_key:
                existing = self._existing_idempotent_job(job_type, idempotency_key)
                if existing is not None:
                    return existing, True
            job = JobRecord(
                job_id=f"job_{uuid.uuid4().hex[:12]}",
                type=job_type,
                status="queued",
                created_at=now,
                updated_at=now,
                progress=progress_record("queued", "Queued.", 0.0),
                metadata=dict(metadata or {}),
                idempotency_key=idempotency_key,
            )
            job.progress_events.append(job.progress)
            self._jobs[job.job_id] = job
            if idempotency_key:
                self._idempotency_index[(job_type, idempotency_key)] = job.job_id
        self._executor.submit(self._run, job.job_id, fn)
        return job, False

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job_id: str, fn: JobCallable) -> None:
        self._update(job_id, status="running", stage="starting", message="Starting job.", percent=1.0)
        try:
            result = fn(lambda stage, message, percent=None, details=None: self.progress(job_id, stage, message, percent, details))
        except Exception as exc:
            error_message = format_job_error(exc)
            self._update(
                job_id,
                status="failed",
                stage="failed",
                message=error_message,
                percent=None,
                error=error_message,
            )
            return
        self._update(job_id, status="succeeded", stage="complete", message="Job complete.", percent=100.0, result=result)

    def progress(
        self,
        job_id: str,
        stage: str,
        message: str,
        percent: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._update(job_id, stage=stage, message=message, percent=percent, details=details)

    def _update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        message: str | None = None,
        percent: float | None = None,
        details: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if status is not None:
                job.status = status
            job.updated_at = utc_now()
            if stage or message or percent is not None or details:
                job.progress = progress_record(
                    stage or str(job.progress.get("stage") or job.status),
                    message or str(job.progress.get("message") or ""),
                    percent if percent is not None else job.progress.get("percent"),
                    details=details,
                )
                job.progress_events.append(job.progress)
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error

    def _existing_idempotent_job(self, job_type: str, idempotency_key: str) -> JobRecord | None:
        job_id = self._idempotency_index.get((job_type, idempotency_key))
        if not job_id:
            return None
        job = self._jobs.get(job_id)
        if job is None:
            self._idempotency_index.pop((job_type, idempotency_key), None)
            return None
        if job.status == "failed":
            self._idempotency_index.pop((job_type, idempotency_key), None)
            return None
        return job


def progress_record(
    stage: str,
    message: str,
    percent: float | None,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "stage": stage,
        "message": message,
        "percent": None if percent is None else max(0.0, min(100.0, float(percent))),
        "updated_at": utc_now(),
    }
    if details:
        record["details"] = dict(details)
    return record


def format_job_error(exc: Exception) -> str:
    message = str(exc).strip()
    if isinstance(exc, RuntimeError) and message:
        return message
    if message:
        return f"{exc.__class__.__name__}: {message}"
    return exc.__class__.__name__


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()
