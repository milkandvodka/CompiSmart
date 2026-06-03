from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request


@dataclass(frozen=True)
class RouteLimit:
    limit: int
    window_seconds: int


DEFAULT_RATE_LIMITS = {
    "default": RouteLimit(limit=120, window_seconds=60),
    "chat": RouteLimit(limit=20, window_seconds=60),
    "jobs": RouteLimit(limit=8, window_seconds=60),
    "job_status": RouteLimit(limit=180, window_seconds=60),
}


EXEMPT_PREFIXES = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
)


def should_rate_limit(request: Request) -> bool:
    if request.method.upper() == "OPTIONS":
        return False
    return not any(request.url.path.startswith(prefix) for prefix in EXEMPT_PREFIXES)


def route_limit_for_path(path: str, limits: dict[str, RouteLimit], *, method: str = "GET") -> RouteLimit:
    if path.startswith("/chat"):
        return limits.get("chat", DEFAULT_RATE_LIMITS["chat"])
    if method.upper() == "GET" and path.startswith("/jobs/"):
        return limits.get("job_status", DEFAULT_RATE_LIMITS["job_status"])
    if path.startswith("/jobs"):
        return limits.get("jobs", DEFAULT_RATE_LIMITS["jobs"])
    return limits.get("default", DEFAULT_RATE_LIMITS["default"])


def client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        host = forwarded_for.split(",", 1)[0].strip()
    else:
        host = request.client.host if request.client else "unknown"
    return f"{host}:{request.method}:{request.url.path}"
