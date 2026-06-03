import unittest
from unittest.mock import Mock

from comparag.vector_store import ChromaChunkStore
from comparag.vector_store import build_where_filter, unpack_query_result
from comparag.models import RagChunk


class ComparagVectorStoreTests(unittest.TestCase):
    def test_build_where_filter_uses_and_for_multiple_filters(self):
        where = build_where_filter(
            comparison_id="demo",
            video_id="A",
            doc_types=["hook_0_5s", "hook_0_10s"],
        )

        self.assertEqual(
            where,
            {
                "$and": [
                    {"comparison_id": {"$eq": "demo"}},
                    {"video_id": {"$eq": "A"}},
                    {"doc_type": {"$in": ["hook_0_5s", "hook_0_10s"]}},
                ]
            },
        )

    def test_unpack_query_result_flattens_chroma_shape(self):
        chunks = unpack_query_result(
            {
                "ids": [["1"]],
                "documents": [["hello"]],
                "metadatas": [[{"video_id": "A"}]],
                "distances": [[0.25]],
            }
        )

        self.assertEqual(chunks[0].id, "1")
        self.assertEqual(chunks[0].text, "hello")
        self.assertEqual(chunks[0].metadata["video_id"], "A")
        self.assertEqual(chunks[0].distance, 0.25)

    def test_delete_chunks_batches_ids(self):
        store = object.__new__(ChromaChunkStore)
        store.collection = Mock()

        store.delete_chunks(["a", "b", "c"], batch_size=2)

        store.collection.delete.assert_any_call(ids=["a", "b"])
        store.collection.delete.assert_any_call(ids=["c"])

    def test_upsert_chunks_skips_empty_batches(self):
        store = object.__new__(ChromaChunkStore)
        store.collection = Mock()

        store.upsert_chunks([])

        store.collection.upsert.assert_not_called()

    def test_upsert_chunks_reports_batch_progress(self):
        store = object.__new__(ChromaChunkStore)
        store.collection = Mock()
        chunks = [
            RagChunk(
                id=f"chunk_{index}",
                comparison_id="demo",
                video_id="A",
                doc_type="full_transcript",
                text="hello",
                display_text="hello",
                metadata={},
            )
            for index in range(3)
        ]
        progress = []

        store.upsert_chunks(chunks, batch_size=2, progress=lambda done, total: progress.append((done, total)))

        self.assertEqual(progress, [(2, 3), (3, 3)])


if __name__ == "__main__":
    unittest.main()
