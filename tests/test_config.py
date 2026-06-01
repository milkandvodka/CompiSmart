import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from comparag.config import codex_testing_enabled, env_flag, get_gemini_api_key, get_llm_mode, load_dotenv_file


class ConfigTests(unittest.TestCase):
    def test_get_gemini_api_key_accepts_aliases(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "alias-key"}, clear=True):
            self.assertEqual(get_gemini_api_key(), "alias-key")

    def test_explicit_key_wins(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}, clear=True):
            self.assertEqual(get_gemini_api_key("explicit-key"), "explicit-key")

    def test_load_dotenv_file_sets_missing_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("GEMINI_API_KEY=from-env-file\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                load_dotenv_file(env_path)
                self.assertEqual(os.environ.get("GEMINI_API_KEY"), "from-env-file")

    def test_codex_testing_mode_env(self):
        with patch.dict(os.environ, {"COMPARAG_LLM_MODE": "codex_testing"}, clear=True):
            self.assertEqual(get_llm_mode(), "codex_testing")
            self.assertTrue(codex_testing_enabled())

    def test_codex_testing_boolean_env(self):
        with patch.dict(os.environ, {"CODEX_TESTING": "true"}, clear=True):
            self.assertTrue(codex_testing_enabled())

    def test_env_flag_accepts_booleanish_values(self):
        with patch.dict(os.environ, {"COMPARAG_DISABLE_GEMINI": "1"}, clear=True):
            self.assertTrue(env_flag("COMPARAG_DISABLE_GEMINI"))


if __name__ == "__main__":
    unittest.main()
