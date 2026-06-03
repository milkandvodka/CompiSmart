import json
import unittest

from comparag.context import DEFAULT_CONTEXT_PROFILE
from comparag.memory import InMemoryConversationMemory
from comparag.models import RetrievedChunk
from comparag.rag_graph import (
    MAX_RETRIEVED_CONTEXT_CHARS,
    RagChatEngine,
    build_prompt,
    fallback_evidence_plan,
    fallback_evidence_plan_for_route,
    fallback_route_question,
    plan_question_with_llm,
    format_retrieved,
    retrieve_for_evidence_plan,
    retrieve_for_route,
    retrieval_questions,
    validate_answer_citations,
)


class FakeRetriever:
    def __init__(self):
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
        return [
            RetrievedChunk(
                id=f"{video_id or 'all'}_{len(self.calls)}",
                text="chunk",
                metadata={"video_id": video_id or "A", "doc_type": (doc_types or ["general"])[0]},
            )
        ]


class EchoPromptLLM:
    def __init__(self):
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if "Return strict JSON only. Do not answer the user." in prompt:
            route = "comments" if "commented" in prompt.lower() else "metrics"
            return json.dumps(
                {
                    "route": route,
                    "needs_structured_metrics": route == "metrics",
                    "balanced_retrieval": False,
                    "doc_types": [],
                    "use_comment_fact_tool": route == "comments",
                    "tool_only": route == "comments",
                    "reason": "test plan",
                }
            )
        if "maintaining long-term memory" in prompt:
            return "User wants follow-up answers to remember Video B focus."
        return "LLM answered with exact tool context [Video B, comment 1]."

    def stream(self, prompt):
        yield self.complete(prompt)


class ComparagRagGraphTests(unittest.TestCase):
    def test_fallback_route_question_identifies_structured_and_hook_questions(self):
        self.assertEqual(fallback_route_question("What's the engagement rate of each?"), "metrics")
        self.assertEqual(fallback_route_question("Who's the creator of Video B?"), "creator")
        self.assertEqual(fallback_route_question("Does Video A prove the creator changed someone's life?"), "general")
        self.assertEqual(fallback_route_question("Compare the hooks in the first 5 seconds"), "hook")
        self.assertEqual(fallback_route_question("Suggest improvements for B"), "improvement")
        self.assertEqual(fallback_route_question("Show top comments with comment likes"), "comments")
        self.assertEqual(fallback_route_question("Which comment themes are noisy?"), "comments")
        self.assertEqual(fallback_route_question("What is the duration of Video B?"), "metrics")
        self.assertEqual(
            fallback_route_question(
                "What were their total likes again?",
                [{"role": "assistant", "content": "Phrase 'beta code': 11 matching comments, 7 total comment likes."}],
            ),
            "comments",
        )

    def test_hook_retrieval_is_balanced_across_a_and_b(self):
        retriever = FakeRetriever()

        chunks = retrieve_for_route(
            retriever=retriever,
            comparison_id="demo",
            question="Compare hooks",
            route="hook",
        )

        self.assertEqual([call["video_id"] for call in retriever.calls], ["A", "B"])
        self.assertEqual(retriever.calls[0]["doc_types"], ["hook_0_5s", "hook_0_10s", "creative_features"])
        self.assertEqual(len(chunks), 2)

    def test_evidence_plan_for_improvements_uses_processed_comment_and_creative_chunks(self):
        plan = fallback_evidence_plan_for_route("Suggest improvements for B", "improvement")

        self.assertTrue(plan["balanced_retrieval"])
        self.assertIn("creative_features", plan["doc_types"])
        self.assertIn("comment_theme", plan["doc_types"])
        self.assertNotIn("top_comments", plan["doc_types"])

    def test_evidence_plan_for_comment_questions_uses_top_comments(self):
        plan = fallback_evidence_plan_for_route("Show top comments for Video B", "comments")

        self.assertFalse(plan["balanced_retrieval"])
        self.assertEqual(plan["video_id"], "B")
        self.assertIn("top_comments", plan["doc_types"])
        self.assertIn("comment_intelligence_summary", plan["doc_types"])

    def test_llm_planner_controls_evidence_plan(self):
        llm = EchoPromptLLM()

        plan = plan_question_with_llm(
            question="Who commented 'beta code' in the insta video?",
            history=[],
            profiles=[],
            llm=llm,
            context_profile=DEFAULT_CONTEXT_PROFILE,
        )

        self.assertEqual(plan["planner_mode"], "llm")
        self.assertEqual(plan["route"], "comments")
        self.assertTrue(plan["use_comment_fact_tool"])

    def test_llm_planner_can_decline_exact_comment_tool(self):
        class PlannerLLM(EchoPromptLLM):
            def complete(self, prompt):
                self.prompts.append(prompt)
                return json.dumps(
                    {
                        "route": "comments",
                        "needs_structured_metrics": False,
                        "balanced_retrieval": False,
                        "doc_types": [],
                        "use_comment_fact_tool": False,
                        "tool_only": False,
                        "reason": "test",
                    }
                )

        llm = PlannerLLM()

        plan = plan_question_with_llm(
            question="Show Instagram comments with usernames, user ids, and profile URLs.",
            history=[],
            profiles=[],
            llm=llm,
            context_profile=DEFAULT_CONTEXT_PROFILE,
        )

        self.assertEqual(plan["route"], "comments")
        self.assertFalse(plan["use_comment_fact_tool"])
        self.assertFalse(plan["tool_only"])

    def test_fallback_planner_uses_exact_comment_tool_for_exact_comment_fields(self):
        plan = fallback_evidence_plan(
            "Show Instagram comments with usernames, user ids, and profile URLs.",
            history=[],
            planner_error="planner failed",
        )

        self.assertEqual(plan["route"], "comments")
        self.assertTrue(plan["use_comment_fact_tool"])
        self.assertTrue(plan["tool_only"])

    def test_entity_name_lookup_retrieves_transcripts_and_comment_chunks_without_exact_comment_tool(self):
        plan = fallback_evidence_plan(
            "What is the name of the series and what was the name of his son?",
            history=[],
            planner_error="planner failed",
        )

        self.assertEqual(plan["route"], "general")
        self.assertTrue(plan["balanced_retrieval"])
        self.assertFalse(plan["use_comment_fact_tool"])
        self.assertFalse(plan["tool_only"])
        self.assertIn("transcript_window", plan["doc_types"])
        self.assertIn("top_comments", plan["doc_types"])
        self.assertIn("comment_intelligence_summary", plan["doc_types"])
        self.assertGreaterEqual(plan["n_results"], 12)

    def test_llm_entity_lookup_plan_is_expanded_to_comment_evidence(self):
        class PlannerLLM(EchoPromptLLM):
            def complete(self, prompt):
                self.prompts.append(prompt)
                return json.dumps(
                    {
                        "route": "general",
                        "needs_structured_metrics": False,
                        "balanced_retrieval": False,
                        "doc_types": ["transcript_window"],
                        "use_comment_fact_tool": False,
                        "tool_only": False,
                        "reason": "test",
                    }
                )

        plan = plan_question_with_llm(
            question="Which series is this and who was the son?",
            history=[],
            profiles=[],
            llm=PlannerLLM(),
            context_profile=DEFAULT_CONTEXT_PROFILE,
        )

        self.assertEqual(plan["planner_mode"], "llm")
        self.assertFalse(plan["use_comment_fact_tool"])
        self.assertFalse(plan["tool_only"])
        self.assertTrue(plan["balanced_retrieval"])
        self.assertIn("transcript_window", plan["doc_types"])
        self.assertIn("top_comments", plan["doc_types"])
        self.assertIn("comment_theme", plan["doc_types"])

    def test_broad_comment_plan_does_not_force_exact_comment_tool(self):
        class PlannerLLM(EchoPromptLLM):
            def complete(self, prompt):
                self.prompts.append(prompt)
                return json.dumps(
                    {
                        "route": "comments",
                        "needs_structured_metrics": False,
                        "balanced_retrieval": False,
                        "doc_types": ["top_comments", "comment_theme"],
                        "use_comment_fact_tool": True,
                        "tool_only": False,
                        "reason": "test",
                    }
                )

        plan = plan_question_with_llm(
            question="Tell me about the least response time algorithm and also tell me about comments.",
            history=[],
            profiles=[],
            llm=PlannerLLM(),
            context_profile=DEFAULT_CONTEXT_PROFILE,
        )

        self.assertEqual(plan["route"], "comments")
        self.assertFalse(plan["use_comment_fact_tool"])
        self.assertFalse(plan["tool_only"])
        self.assertTrue(plan["needs_structured_metrics"])
        self.assertIn("transcript_text_window", plan["doc_types"])
        self.assertIn("comment_intelligence_summary", plan["doc_types"])

    def test_memory_summary_is_loaded_and_updated_after_threshold(self):
        llm = EchoPromptLLM()
        memory = InMemoryConversationMemory()
        memory.append_turn("thread", user="Focus on Video B", assistant="Okay.")
        memory.save_summary("thread", "Existing long-term summary.", {"message_count": 0})
        profile = DEFAULT_CONTEXT_PROFILE.__class__(
            **{
                **DEFAULT_CONTEXT_PROFILE.__dict__,
                "memory_summary_trigger_messages": 2,
                "memory_recent_messages_for_summary": 4,
            }
        )
        engine = RagChatEngine(
            retriever=FakeRetriever(),
            profiles=[{"video_id": "B", "engagement_rate": 2.5, "likes": 10, "comments": 5, "views": 600}],
            context_profile=profile,
            llm=llm,
            memory=memory,
        )

        result = engine.invoke(comparison_id="demo", question="What about its engagement?", thread_id="thread")

        self.assertTrue(result["memory_summary_update"]["updated"])
        self.assertIn("Existing long-term summary.", llm.prompts[-1])
        self.assertIn("User wants follow-up", memory.get_summary("thread")["summary"])

    def test_metrics_route_skips_vector_retrieval(self):
        retriever = FakeRetriever()

        chunks = retrieve_for_route(
            retriever=retriever,
            comparison_id="demo",
            question="engagement rate",
            route="metrics",
        )

        self.assertEqual(chunks, [])
        self.assertEqual(retriever.calls, [])

    def test_comment_tool_answer_skips_vector_retrieval_in_engine(self):
        llm = EchoPromptLLM()
        engine = RagChatEngine(
            retriever=FakeRetriever(),
            profiles=[],
            comment_facts={
                "facts": [
                    {
                        "video_id": "B",
                        "platform": "instagram_post",
                        "comment_id": "1",
                        "text": "Beta code",
                        "normalized_text": "beta code",
                        "author_username": "buyer",
                        "author_id": "u1",
                        "author_url": "https://www.instagram.com/buyer/",
                        "like_count": 2,
                    }
                ]
            },
            llm=llm,
        )

        result = engine.invoke(comparison_id="demo", question="Who commented 'beta code' in the insta video?")

        self.assertIn("LLM answered", result["answer"])
        self.assertIn("1 matching comments", llm.prompts[-1])
        self.assertEqual(result["citations"][0]["label"], "Video B, comment 1")

    def test_format_retrieved_respects_context_budget(self):
        chunks = [
            {
                "id": f"chunk-{index}",
                "text": "x" * 3000,
                "metadata": {"citation_label": f"Video A, chunk {index}"},
            }
            for index in range(10)
        ]

        text = format_retrieved(chunks)

        self.assertLessEqual(len(text), MAX_RETRIEVED_CONTEXT_CHARS + 20)
        self.assertIn("Video A, chunk 0", text)

    def test_validate_answer_citations_flags_invalid_labels(self):
        audit = validate_answer_citations(
            "Use this [Video A, transcript]. Avoid that [Video B, imaginary].",
            [{"label": "Video A, transcript"}],
        )

        self.assertFalse(audit["valid"])
        self.assertEqual(audit["invalid_labels"], ["Video B, imaginary"])

    def test_validate_answer_citations_ignores_non_source_brackets(self):
        audit = validate_answer_citations(
            "Evidence mentions tags ['Vitamin C'] and cites [Video A, transcript].",
            [{"label": "Video A, transcript"}],
        )

        self.assertTrue(audit["valid"])
        self.assertEqual(audit["cited_labels"], ["Video A, transcript"])

    def test_validate_answer_citations_splits_combined_video_labels(self):
        audit = validate_answer_citations(
            "Good evidence [Video A, summary, Video B, theme].",
            [{"label": "Video A, summary"}, {"label": "Video B, theme"}],
        )

        self.assertTrue(audit["valid"])
        self.assertEqual(audit["cited_labels"], ["Video A, summary", "Video B, theme"])

    def test_real_llm_is_called_for_metric_route(self):
        llm = EchoPromptLLM()
        engine = RagChatEngine(
            retriever=FakeRetriever(),
            profiles=[{"video_id": "A", "engagement_rate": 2.5, "likes": 10, "comments": 5, "views": 600}],
            llm=llm,
        )

        result = engine.invoke(comparison_id="demo", question="What's the engagement rate?")

        self.assertIn("LLM answered", result["answer"])
        self.assertIn("engagement_rate: 2.50%", llm.prompts[-1])
        self.assertGreaterEqual(len(llm.prompts), 2)

    def test_prompt_pins_engagement_formula_to_views(self):
        prompt = build_prompt(
            {
                "route": "comparison",
                "question": "Why did A win?",
                "profiles": [
                    {
                        "video_id": "A",
                        "views": 100,
                        "likes": 10,
                        "comments": 5,
                        "engagement_rate": 15,
                        "thumbnail": "https://example.test/thumb.jpg",
                        "creator_url": "https://example.test/creator",
                    }
                ],
                "retrieved": [],
                "history": [],
                "evidence_plan": {},
            },
            [],
        )

        self.assertIn("(likes + comments) / views * 100", prompt)
        self.assertIn("Never use follower count as the denominator", prompt)
        self.assertIn("thumbnail: https://example.test/thumb.jpg", prompt)
        self.assertIn("creator_url: https://example.test/creator", prompt)
        self.assertIn("Do not use outside/world knowledge", prompt)
        self.assertIn("do not replace it with a generic explanation", prompt)
        self.assertIn("Put only one citation label inside each bracket", prompt)
        self.assertIn("checked retrieved transcript, comment, and metadata evidence", prompt)
        self.assertIn("not enough context", prompt)
        self.assertIn("Final grounding checklist", prompt)
        self.assertIn("Never answer a compound question with one blanket", prompt)
        self.assertIn("closest retrieved evidence", prompt)
        self.assertIn("answer the supported part with citations", prompt)

    def test_format_retrieved_includes_transcript_variants_for_ambiguous_asr(self):
        block = format_retrieved(
            [
                {
                    "id": "demo_A_transcript_1",
                    "text": "Video A transcript.\nThe route is from one city to another.",
                    "metadata": {
                        "doc_type": "transcript_window",
                        "citation_label": "Video A, transcript 00:00-00:10",
                        "hinglish_text": "route ek city se doosri city tak hai",
                        "raw_text": "\u0930\u0942\u091f \u090f\u0915 \u0936\u0939\u0930 \u0938\u0947 \u0926\u0942\u0938\u0930\u0947 \u0936\u0939\u0930 \u0924\u0915 \u0939\u0948",
                    },
                }
            ]
        )

        self.assertIn("Hinglish/raw-latin variant", block)
        self.assertIn("Original ASR transcript", block)

    def test_multi_part_questions_retrieve_each_subquestion(self):
        retriever = FakeRetriever()
        plan = {
            "doc_types": ["full_transcript", "top_comments"],
            "n_results": 4,
            "balanced_retrieval": False,
        }

        retrieve_for_evidence_plan(
            retriever=retriever,
            comparison_id="demo",
            question="Explain the algorithm? Also what distance is mentioned? Also summarize comments.",
            evidence_plan=plan,
        )

        query_texts = [call["query_text"] for call in retriever.calls]
        self.assertGreaterEqual(len(query_texts), 3)
        self.assertIn("what distance is mentioned", [query.lower() for query in query_texts])
        self.assertEqual(
            retrieval_questions("Explain topic one? Also explain topic two? Also summarize comments.")[:3],
            [
                "Explain topic one? Also explain topic two? Also summarize comments.",
                "Explain topic one",
                "explain topic two",
            ],
        )


if __name__ == "__main__":
    unittest.main()
