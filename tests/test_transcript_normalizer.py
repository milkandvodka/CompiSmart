import unittest
from unittest.mock import patch

from transcript_normalizer import detect_scripts, normalize_result_payload


class TranscriptNormalizerTests(unittest.TestCase):
    def test_detect_scripts_finds_devanagari_and_arabic_urdu(self):
        scripts = detect_scripts("latin \u0905 \u0627")

        self.assertEqual(scripts, ["devanagari", "arabic_urdu"])

    def test_english_only_transcript_skips_llm(self):
        payload = {"videos": [{"transcript": {"text": "plain latin transcript fixture"}}]}

        normalized = normalize_result_payload(payload, api_key="unused", model="gemini-test", timeout_seconds=1)
        variants = normalized["videos"][0]["transcript"]["variants"]

        self.assertFalse(variants["llm_used"])
        self.assertIsNone(variants["model"])
        self.assertEqual(variants["detected_scripts"], [])
        self.assertEqual(variants["hinglish_latin"], "plain latin transcript fixture")
        self.assertEqual(variants["english_normalized"], "plain latin transcript fixture")

    def test_non_english_script_without_key_adds_note(self):
        payload = {"videos": [{"transcript": {"text": "latin \u0905 fixture"}}]}

        normalized = normalize_result_payload(payload, api_key=None, model="gemini-test", timeout_seconds=1)
        variants = normalized["videos"][0]["transcript"]["variants"]

        self.assertFalse(variants["llm_used"])
        self.assertEqual(variants["detected_scripts"], ["devanagari"])
        self.assertIn("GEMINI_API_KEY", variants["notes"][0])

    def test_non_english_asr_language_without_native_script_uses_completion_provider(self):
        prompts = []
        payload = {
            "videos": [
                {
                    "title": "Synthetic metadata title",
                    "transcript": {
                        "text": "latin_source_token_one latin_source_token_two",
                        "kind": "asr",
                        "language": "zz",
                    },
                }
            ]
        }

        def complete(prompt):
            prompts.append(prompt)
            return """{"detected_language_labels":["Synthetic"],"hinglish_latin":"latin_output_token","english_normalized":"normalized output token","notes":[]}"""

        normalized = normalize_result_payload(
            payload,
            api_key=None,
            model="gemini-test",
            timeout_seconds=1,
            complete=complete,
            provider_label="test-provider",
        )
        variants = normalized["videos"][0]["transcript"]["variants"]

        self.assertTrue(variants["llm_used"])
        self.assertEqual(variants["source_language"], "zz")
        self.assertEqual(variants["detected_scripts"], [])
        self.assertEqual(variants["english_normalized"], "normalized output token")
        self.assertIn("Synthetic metadata title", prompts[0])

    def test_non_english_asr_language_without_key_adds_note(self):
        payload = {
            "videos": [
                {
                    "transcript": {
                        "text": "latin_source_token_one latin_source_token_two",
                        "kind": "asr",
                        "language": "zz",
                    }
                }
            ]
        }

        normalized = normalize_result_payload(payload, api_key=None, model="gemini-test", timeout_seconds=1)
        variants = normalized["videos"][0]["transcript"]["variants"]

        self.assertFalse(variants["llm_used"])
        self.assertEqual(variants["source_language"], "zz")
        self.assertIn("language 'zz' needs normalization", variants["notes"][0])

    @patch("transcript_normalizer.normalize_with_gemini")
    def test_non_english_script_uses_gemini_when_key_is_available(self, gemini_mock):
        gemini_mock.return_value = {
            "detected_language_labels": ["Synthetic"],
            "hinglish_latin": "latin output token",
            "english_normalized": "normalized output token",
            "notes": [],
        }
        payload = {"videos": [{"transcript": {"text": "latin \u0905 fixture"}}]}

        normalized = normalize_result_payload(payload, api_key="key", model="gemini-test", timeout_seconds=1)
        variants = normalized["videos"][0]["transcript"]["variants"]

        self.assertTrue(variants["llm_used"])
        self.assertEqual(variants["model"], "gemini-test")
        self.assertEqual(variants["hinglish_latin"], "latin output token")
        self.assertEqual(variants["english_normalized"], "normalized output token")
        gemini_mock.assert_called_once()

    def test_non_english_script_can_use_generic_completion_provider(self):
        prompts = []
        payload = {
            "videos": [
                {
                    "title": "Synthetic title",
                    "hashtags": ["SyntheticTag"],
                    "transcript": {"text": "latin \u0924 fixture"},
                }
            ]
        }

        def complete(prompt):
            prompts.append(prompt)
            return """```json
{"detected_language_labels":["Synthetic"],"hinglish_latin":"latin output token","english_normalized":"normalized output token","notes":[]}
```"""

        normalized = normalize_result_payload(
            payload,
            api_key=None,
            model="gemini-test",
            timeout_seconds=1,
            complete=complete,
            provider_label="test-provider",
        )
        variants = normalized["videos"][0]["transcript"]["variants"]

        self.assertTrue(variants["llm_used"])
        self.assertEqual(variants["model"], "test-provider")
        self.assertEqual(variants["hinglish_latin"], "latin output token")
        self.assertEqual(variants["english_normalized"], "normalized output token")
        self.assertIn("Synthetic title", prompts[0])
        self.assertIn("SyntheticTag", prompts[0])


if __name__ == "__main__":
    unittest.main()
