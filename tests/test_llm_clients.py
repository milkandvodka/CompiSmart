import unittest
from unittest.mock import Mock, patch

from comparag.analysis_llm import post_json_with_retries
from pathlib import Path

from comparag.llm import CodexTestingLLM, GeminiChatLLM, OpenAIChatLLM, ProviderFallbackLLM, build_codex_testing_prompt


class LlmClientTests(unittest.TestCase):
    def test_gemini_chat_retries_429_before_returning_success(self):
        first = Mock(status_code=429, headers={})
        second = Mock(status_code=200, headers={})
        with patch("comparag.llm.requests.post", side_effect=[first, second]) as post:
            with patch("comparag.llm.time.sleep") as sleep:
                llm = GeminiChatLLM(api_key="key", max_retries=1, retry_backoff_seconds=0.1)
                response = llm.post_stream_with_retries("https://example.test", "prompt")

        self.assertIs(response, second)
        self.assertEqual(post.call_count, 2)
        first.close.assert_called_once()
        sleep.assert_called_once()

    def test_gemini_chat_reads_timeout_and_retry_env(self):
        with patch.dict("comparag.llm.os.environ", {"GEMINI_TIMEOUT_SECONDS": "12.5", "GEMINI_MAX_RETRIES": "0"}):
            llm = GeminiChatLLM(api_key="key")

        self.assertEqual(llm.timeout_seconds, 12.5)
        self.assertEqual(llm.max_retries, 0)

    def test_gemini_json_retries_429_before_returning_success(self):
        first = Mock(status_code=429, headers={})
        second = Mock(status_code=200, headers={})
        with patch("comparag.analysis_llm.requests.post", side_effect=[first, second]) as post:
            with patch("comparag.analysis_llm.time.sleep") as sleep:
                response = post_json_with_retries(
                    "https://example.test",
                    api_key="key",
                    prompt="prompt",
                    timeout_seconds=1,
                    max_retries=1,
                    retry_backoff_seconds=0.1,
                )

        self.assertIs(response, second)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()

    def test_openai_chat_stream_parses_delta_content(self):
        response = Mock(status_code=200, headers={})
        response.raise_for_status = Mock()
        response.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":"hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]
        with patch("comparag.llm.requests.post", return_value=response) as post:
            llm = OpenAIChatLLM(api_key="key", model="gpt-4o-mini", timeout_seconds=1, max_retries=0)
            self.assertEqual(llm.complete("prompt"), "hello world")

        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer key")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "gpt-4o-mini")

    def test_provider_fallback_uses_second_provider_when_first_fails(self):
        first = Mock()
        first.stream.side_effect = RuntimeError("quota")
        second = Mock()
        second.stream.return_value = iter(["ok"])
        llm = ProviderFallbackLLM([("first", first), ("second", second)])

        self.assertEqual(llm.complete("prompt"), "ok")
        first.stream.assert_called_once()
        second.stream.assert_called_once()

    def test_codex_testing_prompt_prevents_tool_use_and_preserves_rag_prompt(self):
        prompt = build_codex_testing_prompt("Available citation labels:\n- Video A, metadata snapshot")

        self.assertIn("Do not run shell commands", prompt)
        self.assertIn("Available citation labels", prompt)

    def test_codex_testing_llm_reads_output_last_message(self):
        def fake_run(command, **kwargs):
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text("codex answer", encoding="utf-8")
            return Mock(returncode=0, stdout="", stderr="")

        with patch("comparag.llm.subprocess.run", side_effect=fake_run) as run:
            llm = CodexTestingLLM(command="codex.cmd", model="test-model", timeout_seconds=1, cwd=".")
            self.assertEqual(llm.complete("prompt"), "codex answer")

        command = run.call_args.args[0]
        self.assertIn("exec", command)
        self.assertIn("--ephemeral", command)
        self.assertIn("--ignore-rules", command)
        self.assertIn("test-model", command)

    def test_codex_testing_llm_defaults_to_empty_temp_workspace(self):
        observed = {}

        def fake_run(command, **kwargs):
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text("codex answer", encoding="utf-8")
            run_dir = Path(command[command.index("--cd") + 1])
            observed["run_dir"] = run_dir
            observed["cwd"] = Path(kwargs["cwd"])
            return Mock(returncode=0, stdout="", stderr="")

        with patch("comparag.llm.subprocess.run", side_effect=fake_run):
            llm = CodexTestingLLM(command="codex.cmd", model="", timeout_seconds=1)
            self.assertEqual(llm.complete("prompt"), "codex answer")

        self.assertEqual(observed["run_dir"].name, "empty_workspace")
        self.assertEqual(observed["cwd"], observed["run_dir"])


if __name__ == "__main__":
    unittest.main()
