import unittest

from comparag.models import RetrievedChunk
from comparag.retrieval import HybridRetriever, RetrievalConfig, reciprocal_rank_fusion


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def query(self, query_text, *, comparison_id, video_id=None, doc_types=None, n_results=6):
        self.calls.append(
            {
                "query_text": query_text,
                "comparison_id": comparison_id,
                "video_id": video_id,
                "doc_types": doc_types,
                "n_results": n_results,
            }
        )
        return self.chunks[:n_results]


class ReverseReranker:
    def rerank(self, query, chunks, *, top_n):
        return list(reversed(chunks))[:top_n]


class ComparagRetrievalTests(unittest.TestCase):
    def test_reciprocal_rank_fusion_marks_shared_chunks_as_hybrid(self):
        semantic = [
            RetrievedChunk(id="a", text="a", metadata={}),
            RetrievedChunk(id="b", text="b", metadata={}),
        ]
        lexical = [
            RetrievedChunk(id="b", text="b", metadata={}),
            RetrievedChunk(id="c", text="c", metadata={}),
        ]

        fused = reciprocal_rank_fusion(
            semantic_chunks=semantic,
            lexical_chunks=lexical,
            fusion_k=60,
            semantic_weight=1.0,
            lexical_weight=1.0,
        )

        self.assertEqual(fused[0].id, "b")
        self.assertEqual(fused[0].metadata["retrieval_source"], "hybrid")
        self.assertEqual({chunk.id for chunk in fused}, {"a", "b", "c"})

    def test_hybrid_retriever_queries_both_sources(self):
        semantic = FakeRetriever([RetrievedChunk(id="s", text="semantic", metadata={})])
        lexical = FakeRetriever([RetrievedChunk(id="l", text="lexical", metadata={})])
        retriever = HybridRetriever(semantic_retriever=semantic, lexical_retriever=lexical)

        results = retriever.query("launch", comparison_id="demo", n_results=2)

        self.assertEqual({result.id for result in results}, {"s", "l"})
        self.assertEqual(len(semantic.calls), 1)
        self.assertEqual(len(lexical.calls), 1)

    def test_semantic_mode_skips_lexical(self):
        semantic = FakeRetriever([RetrievedChunk(id="s", text="semantic", metadata={})])
        lexical = FakeRetriever([RetrievedChunk(id="l", text="lexical", metadata={})])
        retriever = HybridRetriever(
            semantic_retriever=semantic,
            lexical_retriever=lexical,
            config=RetrievalConfig(mode="semantic"),
        )

        results = retriever.query("launch", comparison_id="demo", n_results=1)

        self.assertEqual([result.id for result in results], ["s"])
        self.assertEqual(len(semantic.calls), 1)
        self.assertEqual(lexical.calls, [])

    def test_reranker_reorders_candidates(self):
        semantic = FakeRetriever(
            [
                RetrievedChunk(id="a", text="a", metadata={}),
                RetrievedChunk(id="b", text="b", metadata={}),
            ]
        )
        lexical = FakeRetriever([])
        retriever = HybridRetriever(
            semantic_retriever=semantic,
            lexical_retriever=lexical,
            reranker=ReverseReranker(),
            config=RetrievalConfig(mode="semantic", rerank_top_n=2),
        )

        results = retriever.query("launch", comparison_id="demo", n_results=2)

        self.assertEqual([result.id for result in results], ["b", "a"])


if __name__ == "__main__":
    unittest.main()
