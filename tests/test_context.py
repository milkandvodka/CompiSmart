import unittest

from comparag.context import resolve_context_profile


class ContextProfileTests(unittest.TestCase):
    def test_resolve_context_profile_by_provider(self):
        self.assertEqual(resolve_context_profile("gemini").name, "gemini")
        self.assertEqual(resolve_context_profile("codex_testing").name, "codex_testing")

    def test_resolve_context_profile_by_model_name(self):
        self.assertEqual(resolve_context_profile(model="gpt-4.1").name, "openai")
        self.assertEqual(resolve_context_profile(model="gemini-2.5-flash-lite").name, "gemini")
        self.assertEqual(resolve_context_profile(model="small").name, "small")


if __name__ == "__main__":
    unittest.main()
