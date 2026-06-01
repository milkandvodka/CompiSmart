import tempfile
import unittest
from pathlib import Path

from comparag.lexical import BM25LexicalIndex, load_lexical_index, save_lexical_index, tokenize, tokenize_query
from comparag.models import RagChunk


class ComparagLexicalTests(unittest.TestCase):
    def test_tokenize_keeps_hashtags_and_words(self):
        self.assertEqual(tokenize("Alpha Beta #Launch @brand"), ["alpha", "beta", "#launch", "@brand"])

    def test_tokenize_query_removes_generic_question_words(self):
        self.assertEqual(tokenize_query("Which video mentions alpha and beta?"), ["alpha", "beta"])

    def test_bm25_filters_by_video_and_doc_type(self):
        chunks = [
            RagChunk(
                id="a",
                comparison_id="demo",
                video_id="A",
                doc_type="hook_0_5s",
                text="alpha beta launch",
                display_text="",
            ),
            RagChunk(
                id="b",
                comparison_id="demo",
                video_id="B",
                doc_type="top_comments",
                text="operations update",
                display_text="",
            ),
        ]
        index = BM25LexicalIndex.from_chunks(chunks)

        results = index.query(
            "alpha launch",
            comparison_id="demo",
            video_id="A",
            doc_types=["hook_0_5s"],
            n_results=2,
        )

        self.assertEqual([result.id for result in results], ["a"])
        self.assertEqual(results[0].metadata["retrieval_source"], "lexical")

    def test_save_and_load_lexical_index(self):
        chunk = RagChunk(
            id="a",
            comparison_id="demo",
            video_id="A",
            doc_type="full_transcript",
            text="quarterly planning notes",
            display_text="",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            save_lexical_index([chunk], comparison_id="demo", app_dir=Path(temp_dir))
            loaded = load_lexical_index("demo", Path(temp_dir))

        self.assertEqual(loaded.query("planning", comparison_id="demo")[0].id, "a")


if __name__ == "__main__":
    unittest.main()
