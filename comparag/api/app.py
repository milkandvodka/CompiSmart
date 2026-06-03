from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .jobs import JobRegistry
from .protection import DuplicateRequestCache, InMemoryRateLimiter, StreamDuplicateGuard, fingerprint_payload
from .rate_limit import DEFAULT_RATE_LIMITS, RouteLimit, client_key, route_limit_for_path, should_rate_limit
from .schemas import (
    ChatRequest,
    ChatResponse,
    ComparisonListResponse,
    ComparisonSummary,
    ExtractAndIndexRequest,
    IndexComparisonRequest,
    JobResponse,
)
from .services import AgentService, comparison_summary


def create_app(
    *,
    agent_service: AgentService | None = None,
    job_registry: JobRegistry | None = None,
    rate_limiter: InMemoryRateLimiter | None = None,
    rate_limits: dict[str, RouteLimit] | None = None,
    chat_duplicate_cache: DuplicateRequestCache | None = None,
    stream_duplicate_guard: StreamDuplicateGuard | None = None,
) -> FastAPI:
    app = FastAPI(
        title="compaRAG API",
        version="0.1.0",
        description="FastAPI backend around the social-video comparison RAG agent.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.agent_service = agent_service or AgentService()
    app.state.job_registry = job_registry or JobRegistry()
    app.state.rate_limiter = rate_limiter or InMemoryRateLimiter()
    app.state.rate_limits = rate_limits or DEFAULT_RATE_LIMITS
    app.state.chat_duplicate_cache = chat_duplicate_cache or DuplicateRequestCache()
    app.state.stream_duplicate_guard = stream_duplicate_guard or StreamDuplicateGuard()
    mount_frontend(app)

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if not should_rate_limit(request):
            return await call_next(request)
        limit = route_limit_for_path(request.url.path, request.app.state.rate_limits, method=request.method)
        decision = request.app.state.rate_limiter.check(
            client_key(request),
            limit=limit.limit,
            window_seconds=limit.window_seconds,
        )
        if not decision.allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Try again shortly.",
                    "retry_after_seconds": decision.retry_after_seconds,
                },
                headers={
                    "Retry-After": str(decision.retry_after_seconds),
                    "X-RateLimit-Limit": str(decision.limit),
                    "X-RateLimit-Remaining": "0",
                },
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(decision.limit)
        response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/media/thumbnail")
    def thumbnail_proxy(url: str) -> Response:
        if not is_allowed_thumbnail_url(url):
            raise HTTPException(status_code=400, detail="Unsupported thumbnail URL.")
        try:
            upstream = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail="Thumbnail fetch failed.") from exc
        if upstream.status_code >= 400:
            raise HTTPException(status_code=upstream.status_code, detail="Thumbnail fetch failed.")
        content_type = upstream.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Thumbnail URL did not return an image.")
        return Response(
            content=upstream.content,
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.get("/comparisons", response_model=ComparisonListResponse)
    def list_comparisons(request: Request, app_dir: str | None = None) -> dict[str, Any]:
        service = get_service(request)
        return {"comparisons": service.list_comparisons(app_dir=app_dir)}

    @app.get("/comparisons/{comparison_id}", response_model=ComparisonSummary)
    def get_comparison(comparison_id: str, request: Request, app_dir: str | None = None) -> dict[str, Any]:
        service = get_service(request)
        try:
            return comparison_summary(service.get_comparison(comparison_id, app_dir=app_dir))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
        service = get_service(request)
        dedupe_key = request_fingerprint("chat", payload, request)
        try:
            result, deduped = request.app.state.chat_duplicate_cache.run(
                dedupe_key,
                lambda: service.chat(payload),
            )
            result["deduped"] = deduped
            return result
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/chat/stream")
    def stream_chat(payload: ChatRequest, request: Request) -> StreamingResponse:
        service = get_service(request)

        def generate() -> Iterable[str]:
            dedupe_key = request_fingerprint("chat_stream", payload, request)
            guarded = request.app.state.stream_duplicate_guard.stream(
                dedupe_key,
                lambda: safe_stream_chat(service, payload),
            )
            for event in guarded:
                yield encode_sse(event)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/jobs/index", response_model=JobResponse, status_code=202)
    def create_index_job(payload: IndexComparisonRequest, request: Request) -> dict[str, Any]:
        service = get_service(request)
        jobs = get_jobs(request)
        idem = request_fingerprint("job_index", payload, request)
        job, deduped = jobs.submit(
            "index",
            lambda progress: service.index_comparison(payload, progress=progress),
            metadata={"comparison_id": payload.comparison_id},
            idempotency_key=idem,
        )
        return job.to_dict(deduped=deduped)

    @app.post("/jobs/extract-index", response_model=JobResponse, status_code=202)
    def create_extract_index_job(payload: ExtractAndIndexRequest, request: Request) -> dict[str, Any]:
        service = get_service(request)
        jobs = get_jobs(request)
        idem = request_fingerprint("job_extract_index", payload, request)
        job, deduped = jobs.submit(
            "extract-index",
            lambda progress: service.extract_and_index(payload, progress=progress),
            metadata={"comparison_id": payload.comparison_id},
            idempotency_key=idem,
        )
        return job.to_dict(deduped=deduped)

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str, request: Request) -> dict[str, Any]:
        job = get_jobs(request).get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        return job.to_dict()

    return app


def mount_frontend(app: FastAPI) -> None:
    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    static_index = frontend_dir / "static" / "index.html"
    if not static_index.exists():
        return

    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/", include_in_schema=False)
    def ui_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/static/index.html")

    app.mount("/ui", StaticFiles(directory=frontend_dir), name="ui")


def get_service(request: Request) -> AgentService:
    return request.app.state.agent_service


def get_jobs(request: Request) -> JobRegistry:
    return request.app.state.job_registry


def encode_sse(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or "message")
    data = json.dumps(event, ensure_ascii=False)
    return f"event: {event_type}\ndata: {data}\n\n"


def safe_stream_chat(service: AgentService, payload: ChatRequest) -> Iterable[dict[str, Any]]:
    try:
        yield from service.stream_chat(payload)
    except FileNotFoundError as exc:
        yield {"type": "error", "status_code": 404, "detail": str(exc)}
    except ValueError as exc:
        yield {"type": "error", "status_code": 400, "detail": str(exc)}
    except RuntimeError as exc:
        yield {"type": "error", "status_code": 503, "detail": str(exc)}


def request_fingerprint(namespace: str, payload, request: Request) -> str:
    explicit = request.headers.get("Idempotency-Key") or getattr(payload, "idempotency_key", None)
    return fingerprint_payload(
        namespace=namespace,
        payload=payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload,
        explicit_key=explicit,
    )


def is_allowed_thumbnail_url(url: str) -> bool:
    parsed = urlparse(str(url or ""))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    allowed_suffixes = (
        "ytimg.com",
        "youtube.com",
        "googleusercontent.com",
        "cdninstagram.com",
        "fbcdn.net",
        "fbsbx.com",
        "instagram.com",
    )
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in allowed_suffixes)


app = create_app()
