import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from comparag.api.app import create_app, encode_sse
from comparag.api.indexing import build_extractor_command
from comparag.api.jobs import JobRegistry, format_job_error
from comparag.api.rate_limit import RouteLimit
from comparag.api.services import AgentService, summarize_extractor_stderr
from comparag.api.schemas import ExtractAndIndexRequest


class FakeAgentService:
    def __init__(self):
        self.chat_requests = []
        self.stream_requests = []

    def list_comparisons(self, *, app_dir=None):
        return [
            {
                "comparison_id": "demo",
                "chunk_count": 4,
                "embedding_model": "BAAI/bge-m3",
                "collection_name": "comparag_chunks_BAAI_bge-m3",
                "observability_run_id": "run_1",
                "videos": [{"video_id": "A"}, {"video_id": "B"}],
            }
        ]

    def get_comparison(self, comparison_id, *, app_dir=None):
        if comparison_id == "missing":
            raise FileNotFoundError("No comparison record found")
        return self.list_comparisons(app_dir=app_dir)[0]

    def chat(self, request):
        self.chat_requests.append(request)
        return {
            "answer": "Video A wins on engagement [Video A, metadata snapshot].",
            "citations": [{"label": "Video A, metadata snapshot", "video_id": "A", "doc_type": "video_fact_card"}],
            "citation_audit": {
                "allowed_labels": ["Video A, metadata snapshot"],
                "cited_labels": ["Video A, metadata snapshot"],
                "invalid_labels": [],
                "valid": True,
            },
            "evidence_plan": {"route": "metrics"},
            "route": "metrics",
            "llm_error": None,
            "memory_summary_update": {"updated": False},
        }

    def stream_chat(self, request):
        self.stream_requests.append(request)
        yield {
            "type": "metadata",
            "route": "metrics",
            "evidence_plan": {"route": "metrics"},
            "citations": [{"label": "Video A, metadata snapshot"}],
        }
        yield {"type": "token", "text": "hello "}
        yield {"type": "done", "citation_audit": {"valid": True}, "llm_error": None}

    def index_comparison(self, request, *, progress=None):
        if progress:
            progress("test_index", "Indexing in fake service.", 50.0, {"comparison_id": request.comparison_id})
        return {"comparison_id": request.comparison_id, "chunk_count": 2}

    def extract_and_index(self, request, *, progress=None):
        if progress:
            progress("test_extract", "Extracting in fake service.", 25.0, {"comparison_id": request.comparison_id})
        return {"comparison_id": request.comparison_id, "index": {"chunk_count": 2}}


class ApiTests(unittest.TestCase):
    def make_client(self):
        service = FakeAgentService()
        app = create_app(agent_service=service, job_registry=JobRegistry(max_workers=1))
        return TestClient(app), service

    def test_health(self):
        client, _ = self.make_client()
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_comparison_routes(self):
        client, _ = self.make_client()
        listing = client.get("/comparisons")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["comparisons"][0]["comparison_id"], "demo")

        detail = client.get("/comparisons/demo")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["videos"][0]["video_id"], "A")

        missing = client.get("/comparisons/missing")
        self.assertEqual(missing.status_code, 404)

    def test_real_service_lists_comparison_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparisons = root / "comparisons"
            comparisons.mkdir()
            (comparisons / "demo.json").write_text(
                json.dumps(
                    {
                        "comparison_id": "demo",
                        "chunks": [{"id": "chunk_1"}],
                        "embedding_model": "BAAI/bge-m3",
                        "collection_name": "comparag_chunks_BAAI_bge-m3",
                        "videos": [{"video_id": "A"}, {"video_id": "B"}],
                    }
                ),
                encoding="utf-8",
            )
            app = create_app(agent_service=AgentService(), job_registry=JobRegistry(max_workers=1))
            client = TestClient(app)
            response = client.get("/comparisons", params={"app_dir": str(root)})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["comparisons"][0]["comparison_id"], "demo")

    def test_ui_mount_serves_static_frontend(self):
        client, _ = self.make_client()
        redirect = client.get("/ui", follow_redirects=False)
        self.assertEqual(redirect.status_code, 307)
        self.assertEqual(redirect.headers["location"], "/ui/static/index.html")

        html = client.get("/ui/static/index.html")
        self.assertEqual(html.status_code, 200)
        self.assertIn("compaRAG", html.text)

    def test_thumbnail_proxy_rejects_unsupported_hosts(self):
        client, _ = self.make_client()
        response = client.get("/media/thumbnail", params={"url": "http://127.0.0.1/private.png"})

        self.assertEqual(response.status_code, 400)

    @patch("comparag.api.app.requests.get")
    def test_thumbnail_proxy_fetches_allowed_image_host(self, get_mock):
        class FakeResponse:
            status_code = 200
            content = b"image"
            headers = {"content-type": "image/jpeg"}

        get_mock.return_value = FakeResponse()
        client, _ = self.make_client()
        response = client.get("/media/thumbnail", params={"url": "https://scontent.cdninstagram.com/thumb.jpg"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"image")
        self.assertEqual(response.headers["content-type"], "image/jpeg")

    def test_extractor_stderr_summary_prefers_error_line_and_redacts(self):
        stderr = "\n".join(
            [
                "Traceback (most recent call last):",
                "  File \"/app/social_video_extractor.py\", line 1, in <module>",
                "error: failed with key sk-proj-secretsecretsecretsecretsecret and token hf_secretsecretsecretsecretsecret",
            ]
        )

        summary = summarize_extractor_stderr(stderr)

        self.assertTrue(summary.startswith("error: failed"))
        self.assertIn("[redacted-openai-key]", summary)
        self.assertIn("[redacted-hf-token]", summary)

    def test_extractor_command_uses_default_instagram_cookie_file(self):
        with TemporaryDirectory() as tmp:
            cookie_path = Path(tmp) / "instagram-cookies.txt"
            cookie_path.write_text("cookies", encoding="utf-8")
            request = ExtractAndIndexRequest(
                comparison_id="demo",
                youtube_url="https://www.youtube.com/shorts/fMXq_lMB1Jo",
                instagram_url="https://www.instagram.com/reel/DZB9wNBzvLT/",
            )
            with patch.dict(os.environ, {"INSTAGRAM_COOKIES": str(cookie_path)}, clear=False):
                command = build_extractor_command(request, Path(tmp) / "out.json")

        self.assertIn("--cookies", command)
        self.assertIn(str(cookie_path), command)

    def test_chat_endpoint_delegates_to_agent(self):
        client, service = self.make_client()
        response = client.post(
            "/chat",
            json={
                "comparison_id": "demo",
                "question": "What is the engagement rate?",
                "thread_id": "thread_1",
                "options": {"llm": "fallback", "memory_backend": "local"},
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["route"], "metrics")
        self.assertTrue(payload["citation_audit"]["valid"])
        self.assertEqual(service.chat_requests[0].thread_id, "thread_1")
        self.assertEqual(service.chat_requests[0].options.llm, "fallback")

    def test_chat_endpoint_dedupes_quick_duplicate(self):
        client, service = self.make_client()
        body = {
            "comparison_id": "demo",
            "question": "What is the engagement rate?",
            "thread_id": "thread_1",
            "options": {"llm": "fallback", "memory_backend": "local"},
        }
        first = client.post("/chat", json=body)
        second = client.post("/chat", json=body)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertFalse(first.json()["deduped"])
        self.assertTrue(second.json()["deduped"])
        self.assertEqual(len(service.chat_requests), 1)

    def test_stream_endpoint_emits_sse(self):
        client, service = self.make_client()
        with client.stream(
            "POST",
            "/chat/stream",
            json={"comparison_id": "demo", "question": "stream it"},
        ) as response:
            body = response.read().decode("utf-8")
        self.assertEqual(response.status_code, 200)
        self.assertIn("event: metadata", body)
        self.assertIn('"type": "token"', body)
        self.assertIn("event: done", body)
        self.assertEqual(service.stream_requests[0].comparison_id, "demo")

    def test_stream_endpoint_replays_completed_duplicate_without_second_call(self):
        client, service = self.make_client()
        body = {"comparison_id": "demo", "question": "stream it"}
        first = client.post("/chat/stream", json=body)
        second = client.post("/chat/stream", json=body)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertIn("event: metadata", second.text)
        self.assertEqual(len(service.stream_requests), 1)

    def test_jobs_index_endpoint_runs_background_job(self):
        client, _ = self.make_client()
        created = client.post(
            "/jobs/index",
            json={"comparison_id": "demo", "payload": {"videos": []}, "options": {"comment_intelligence": "off"}},
        )
        self.assertEqual(created.status_code, 202)
        job_id = created.json()["job_id"]
        final = None
        for _ in range(20):
            final = client.get(f"/jobs/{job_id}")
            if final.json()["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)
        self.assertEqual(final.status_code, 200)
        self.assertEqual(final.json()["status"], "succeeded")
        self.assertEqual(final.json()["result"]["comparison_id"], "demo")
        self.assertEqual(final.json()["progress"]["stage"], "complete")
        self.assertTrue(final.json()["progress_events"])

    def test_jobs_index_endpoint_dedupes_identical_payload(self):
        client, service = self.make_client()
        body = {"comparison_id": "demo", "payload": {"videos": []}, "options": {"comment_intelligence": "off"}}
        first = client.post("/jobs/index", json=body)
        second = client.post("/jobs/index", json=body)

        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 202)
        self.assertEqual(first.json()["job_id"], second.json()["job_id"])
        self.assertFalse(first.json()["deduped"])
        self.assertTrue(second.json()["deduped"])
        self.assertIn("stage", second.json()["progress"])
        self.assertEqual(len(service.chat_requests), 0)

    def test_failed_idempotent_job_can_be_retried(self):
        calls = {"count": 0}
        jobs = JobRegistry(max_workers=1)

        def failing_job(progress):
            calls["count"] += 1
            raise RuntimeError("boom")

        first, first_deduped = jobs.submit("index", failing_job, idempotency_key="same")
        for _ in range(20):
            current = jobs.get(first.job_id)
            if current and current.status == "failed":
                break
            time.sleep(0.05)

        second, second_deduped = jobs.submit("index", failing_job, idempotency_key="same")

        self.assertFalse(first_deduped)
        self.assertFalse(second_deduped)
        self.assertNotEqual(first.job_id, second.job_id)
        self.assertGreaterEqual(calls["count"], 1)

    def test_job_error_formatting_omits_runtimeerror_prefix(self):
        self.assertEqual(format_job_error(RuntimeError("Extractor failed: error: blocked")), "Extractor failed: error: blocked")
        self.assertEqual(format_job_error(ValueError("bad input")), "ValueError: bad input")

    def test_rate_limit_blocks_repeated_requests(self):
        service = FakeAgentService()
        app = create_app(
            agent_service=service,
            job_registry=JobRegistry(max_workers=1),
            rate_limits={
                "default": RouteLimit(limit=1, window_seconds=60),
                "chat": RouteLimit(limit=10, window_seconds=60),
                "jobs": RouteLimit(limit=10, window_seconds=60),
            },
        )
        client = TestClient(app)

        self.assertEqual(client.get("/comparisons").status_code, 200)
        blocked = client.get("/comparisons")
        self.assertEqual(blocked.status_code, 429)
        self.assertIn("Retry-After", blocked.headers)

    def test_job_status_polling_has_separate_rate_limit(self):
        service = FakeAgentService()
        app = create_app(
            agent_service=service,
            job_registry=JobRegistry(max_workers=1),
            rate_limits={
                "default": RouteLimit(limit=100, window_seconds=60),
                "jobs": RouteLimit(limit=1, window_seconds=60),
                "job_status": RouteLimit(limit=5, window_seconds=60),
            },
        )
        client = TestClient(app)

        created = client.post(
            "/jobs/index",
            json={"comparison_id": "demo", "payload": {"videos": []}, "options": {"comment_intelligence": "off"}},
        )
        self.assertEqual(created.status_code, 202)
        blocked_create = client.post(
            "/jobs/index",
            json={"comparison_id": "other", "payload": {"videos": []}, "options": {"comment_intelligence": "off"}},
        )
        self.assertEqual(blocked_create.status_code, 429)

        job_id = created.json()["job_id"]
        for _ in range(4):
            status = client.get(f"/jobs/{job_id}")
            self.assertEqual(status.status_code, 200)

    def test_encode_sse_json_payload(self):
        encoded = encode_sse({"type": "token", "text": "hello"})
        self.assertEqual(encoded, 'event: token\ndata: {"type": "token", "text": "hello"}\n\n')


if __name__ == "__main__":
    unittest.main()
