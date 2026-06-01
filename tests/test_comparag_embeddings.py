import unittest

from comparag.embeddings import add_embedding_prefixes, resolve_embedding_model
from comparag.reranking import resolve_reranker_model


class ComparagEmbeddingsTests(unittest.TestCase):
    def test_resolve_embedding_model_supports_presets(self):
        self.assertEqual(resolve_embedding_model("fast"), "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(resolve_embedding_model("balanced"), "intfloat/multilingual-e5-base")
        self.assertEqual(resolve_embedding_model("quality"), "BAAI/bge-m3")
        self.assertEqual(resolve_embedding_model("custom/model"), "custom/model")

    def test_e5_adds_query_and_passage_prefixes(self):
        self.assertEqual(
            add_embedding_prefixes(["sample phrase"], "intfloat/multilingual-e5-base", kind="query"),
            ["query: sample phrase"],
        )
        self.assertEqual(
            add_embedding_prefixes(["sample phrase"], "intfloat/multilingual-e5-base", kind="document"),
            ["passage: sample phrase"],
        )

    def test_bge_does_not_add_e5_prefixes(self):
        self.assertEqual(add_embedding_prefixes(["sample phrase"], "BAAI/bge-m3", kind="query"), ["sample phrase"])

    def test_resolve_reranker_model_supports_presets(self):
        self.assertEqual(resolve_reranker_model("quality"), "BAAI/bge-reranker-base")
        self.assertEqual(resolve_reranker_model("fast"), "cross-encoder/ms-marco-MiniLM-L6-v2")


if __name__ == "__main__":
    unittest.main()
