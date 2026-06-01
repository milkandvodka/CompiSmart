import unittest
from unittest.mock import patch

from transcript_normalizer import detect_scripts, normalize_result_payload


class TranscriptNormalizerTests(unittest.TestCase):
    def test_detect_scripts_finds_devanagari_and_arabic_urdu(self):
        scripts = detect_scripts("Update \u0906\u091c\u093e\u090f\u0917\u093e \u0627\u0648\u0631 results visible honge")

        self.assertEqual(scripts, ["devanagari", "arabic_urdu"])

    def test_english_only_transcript_skips_llm(self):
        payload = {"videos": [{"transcript": {"text": "Use this checklist every morning."}}]}

        normalized = normalize_result_payload(payload, api_key="unused", model="gemini-test", timeout_seconds=1)
        variants = normalized["videos"][0]["transcript"]["variants"]

        self.assertFalse(variants["llm_used"])
        self.assertIsNone(variants["model"])
        self.assertEqual(variants["detected_scripts"], [])
        self.assertEqual(variants["hinglish_latin"], "Use this checklist every morning.")
        self.assertEqual(variants["english_normalized"], "Use this checklist every morning.")

    def test_non_english_script_without_key_adds_note(self):
        payload = {"videos": [{"transcript": {"text": "report \u092a\u0930 update \u0906\u091c\u093e\u090f\u0917\u093e"}}]}

        normalized = normalize_result_payload(payload, api_key=None, model="gemini-test", timeout_seconds=1)
        variants = normalized["videos"][0]["transcript"]["variants"]

        self.assertFalse(variants["llm_used"])
        self.assertEqual(variants["detected_scripts"], ["devanagari"])
        self.assertIn("GEMINI_API_KEY", variants["notes"][0])

    @patch("transcript_normalizer.normalize_with_gemini")
    def test_non_english_script_uses_gemini_when_key_is_available(self, gemini_mock):
        gemini_mock.return_value = {
            "detected_language_labels": ["Hindi", "English", "Hinglish"],
            "hinglish_latin": "report par update aa jayega",
            "english_normalized": "The report will get an update.",
            "notes": [],
        }
        payload = {"videos": [{"transcript": {"text": "report \u092a\u0930 update \u0906\u091c\u093e\u090f\u0917\u093e"}}]}

        normalized = normalize_result_payload(payload, api_key="key", model="gemini-test", timeout_seconds=1)
        variants = normalized["videos"][0]["transcript"]["variants"]

        self.assertTrue(variants["llm_used"])
        self.assertEqual(variants["model"], "gemini-test")
        self.assertEqual(variants["hinglish_latin"], "report par update aa jayega")
        self.assertEqual(variants["english_normalized"], "The report will get an update.")
        gemini_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
