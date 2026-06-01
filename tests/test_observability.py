import unittest

from comparag.observability import scrub_secrets


class ObservabilityTests(unittest.TestCase):
    def test_scrub_secrets_redacts_nested_sensitive_keys(self):
        payload = {
            "api_key": "secret",
            "nested": {"password": "pw", "safe": "ok"},
            "items": [{"token": "abc"}],
        }

        scrubbed = scrub_secrets(payload)

        self.assertEqual(scrubbed["api_key"], "[REDACTED]")
        self.assertEqual(scrubbed["nested"]["password"], "[REDACTED]")
        self.assertEqual(scrubbed["nested"]["safe"], "ok")
        self.assertEqual(scrubbed["items"][0]["token"], "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
