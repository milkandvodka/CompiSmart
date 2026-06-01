import unittest

from comparag.cli import resolve_chat_embedding_model


class ComparagCliTests(unittest.TestCase):
    def test_chat_embedding_model_defaults_to_record_model(self):
        self.assertEqual(
            resolve_chat_embedding_model(None, {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"}),
            "sentence-transformers/all-MiniLM-L6-v2",
        )

    def test_chat_embedding_model_allows_override_preset(self):
        self.assertEqual(resolve_chat_embedding_model("quality", {"embedding_model": "fast"}), "BAAI/bge-m3")


if __name__ == "__main__":
    unittest.main()
