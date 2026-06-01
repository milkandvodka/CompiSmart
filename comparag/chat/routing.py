from __future__ import annotations

import json
import re
from typing import Any

from ..context import ModelContextProfile
from ..llm import ChatLLM, ExtractiveFallbackLLM
from ..memory import Message
from .constants import ALLOWED_DOC_TYPES, ALLOWED_ROUTES, STRUCTURED_ROUTES
from .prompting import format_memory, format_profiles


def plan_question_with_llm(
    *,
    question: str,
    history: list[Message],
    memory_summary: str = "",
    profiles: list[dict[str, Any]],
    llm: ChatLLM | None,
    context_profile: ModelContextProfile,
) -> dict[str, Any]:
    if llm is None or isinstance(llm, ExtractiveFallbackLLM):
        return fallback_evidence_plan(question, history, planner_error="No LLM planner configured.")
    prompt = build_planner_prompt(
        question=question,
        history=history,
        memory_summary=memory_summary,
        profiles=profiles,
        context_profile=context_profile,
    )
    try:
        raw = llm.complete(prompt)
        parsed = parse_json_object(raw)
        return sanitize_evidence_plan(parsed, question=question, history=history)
    except Exception as exc:
        return fallback_evidence_plan(question, history, planner_error=f"{exc.__class__.__name__}: planner failed")


def build_planner_prompt(
    *,
    question: str,
    history: list[Message],
    memory_summary: str,
    profiles: list[dict[str, Any]],
    context_profile: ModelContextProfile,
) -> str:
    return f"""
You are the evidence planner for a two-video creator analytics RAG chatbot.

Return strict JSON only. Do not answer the user.

Choose how the answer should gather evidence:
{{
  "route": "metrics|creator|comments|hook|improvement|comparison|general",
  "needs_structured_metrics": true,
  "balanced_retrieval": false,
  "video_id": "A|B|null",
  "doc_types": ["allowed doc type names"],
  "per_video": 4,
  "n_results": 8,
  "use_comment_fact_tool": false,
  "tool_only": false,
  "reason": "short reason"
}}

Allowed doc_types:
{", ".join(sorted(ALLOWED_DOC_TYPES))}

Use structured metrics for exact views, likes, comments, engagement rate, creator, follower/subscriber count, upload date, duration, hashtags, and thumbnails.
Use the comment fact tool for exact comment objects, commenters, comment-like counts, author URLs, author IDs, exact phrase searches, and top fetched comments.
Use vector retrieval for transcript meaning, hooks, creative features, audience themes, comparisons, and recommendations.
Use balanced retrieval when comparing Video A and Video B.

Conversation memory:
{format_memory(history, memory_summary=memory_summary, context_profile=context_profile)}

Structured video metrics available:
{format_profiles(profiles)}

User question:
{question}
""".strip()


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I).strip()
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if not match:
        raise ValueError("Planner did not return JSON.")
    return json.loads(match.group(0))


def sanitize_evidence_plan(data: dict[str, Any], *, question: str, history: list[Message]) -> dict[str, Any]:
    route = str(data.get("route") or "general").strip().lower()
    if route not in ALLOWED_ROUTES:
        route = "general"
    fallback = fallback_evidence_plan_for_route(question, route, history)
    doc_types = [str(item) for item in data.get("doc_types") or [] if str(item) in ALLOWED_DOC_TYPES]
    needs_comment_facts = route == "comments" and asks_for_comment_fact_tool(question, history)
    plan = {
        **fallback,
        "route": route,
        "needs_structured_metrics": bool(data.get("needs_structured_metrics", fallback.get("needs_structured_metrics"))),
        "balanced_retrieval": bool(data.get("balanced_retrieval", fallback.get("balanced_retrieval"))),
        "doc_types": doc_types or fallback.get("doc_types") or [],
        "reason": str(data.get("reason") or fallback.get("reason") or "LLM evidence plan."),
        "use_comment_fact_tool": bool(data.get("use_comment_fact_tool", fallback.get("use_comment_fact_tool", False)))
        or needs_comment_facts,
        "tool_only": bool(data.get("tool_only", fallback.get("tool_only", False))) or needs_comment_facts,
        "planner_mode": "llm",
    }
    video_id = data.get("video_id")
    if video_id in ("A", "B"):
        plan["video_id"] = video_id
    elif "video_id" in plan:
        plan.pop("video_id", None)
    for key in ("per_video", "n_results"):
        try:
            if data.get(key) is not None:
                plan[key] = max(1, min(20, int(data[key])))
        except (TypeError, ValueError):
            pass
    return plan


def fallback_evidence_plan(
    question: str,
    history: list[Message] | None = None,
    *,
    planner_error: str | None = None,
) -> dict[str, Any]:
    route = fallback_route_question(question, history)
    plan = fallback_evidence_plan_for_route(question, route, history or [])
    plan["planner_mode"] = "fallback"
    if planner_error:
        plan["planner_error"] = planner_error
    return plan


def fallback_route_question(question: str, history: list[Message] | None = None) -> str:
    text = question.lower()
    if asks_for_comment_objects(text):
        return "comments"
    if should_route_to_comments_from_history(text, history or []):
        return "comments"
    if asks_for_creator_metadata(text):
        return "creator"
    if any(
        term in text
        for term in [
            "engagement rate",
            "views",
            "likes",
            "comments",
            "metrics",
            "performance",
            "duration",
            "length",
            "upload",
            "posted",
            "hashtag",
            "thumbnail",
        ]
    ):
        return "metrics"
    if any(term in text for term in ["hook", "first 5", "first five", "opening", "intro", "catchy", "attention"]):
        return "hook"
    if any(term in text for term in ["improve", "improvement", "suggest", "recommend", "fix"]):
        return "improvement"
    if any(term in text for term in ["why", "compare", "more engagement", "better", "worked"]):
        return "comparison"
    return "general"


def should_route_to_comments_from_history(text: str, history: list[Message]) -> bool:
    if not should_resolve_from_history(text):
        return False
    if not any(term in text for term in ["like", "likes", "count", "url", "id", "profile", "who", "they", "their"]):
        return False
    recent = "\n".join(str(message.get("content") or "").lower() for message in history[-4:])
    return "comment" in recent or "commented" in recent or "comment likes" in recent


def asks_for_creator_metadata(text: str) -> bool:
    if any(term in text for term in ["follower", "subscriber", "channel"]):
        return True
    return bool(
        re.search(r"\bwho(?:'s| is| are| was)?\s+(?:the\s+)?creator\b", text)
        or re.search(r"\bcreator\s+(?:of|for)\s+video\b", text)
        or re.search(r"\bwho\s+(?:made|posted|created)\b", text)
    )


def asks_for_comment_objects(text: str) -> bool:
    return any(
        term in text
        for term in [
            "top comment",
            "most liked comment",
            "comment likes",
            "comment-like",
            "commenter",
            "fetched comment",
            "every comment",
            "all comments",
            "list comments",
            "comment theme",
            "who commented",
            "who wrote",
            "username",
            "profile url",
            "user id",
            "audience reaction",
            "noisy comment",
            "useful comment",
        ]
    )


def asks_for_comment_fact_tool(question: str, history: list[Message] | None = None) -> bool:
    text = question.lower()
    if asks_for_comment_objects(text):
        return True
    if any(term in text for term in ["user id", "profile url", "comment-like", "comment like", "total comment likes"]):
        return True
    return should_route_to_comments_from_history(text, history or [])


def fallback_evidence_plan_for_route(question: str, route: str, history: list[Message] | None = None) -> dict[str, Any]:
    if route in STRUCTURED_ROUTES:
        return {
            "route": route,
            "needs_structured_metrics": True,
            "balanced_retrieval": False,
            "doc_types": [],
            "per_video": 0,
            "use_comment_fact_tool": False,
            "tool_only": False,
            "reason": "Exact structured metrics are available; vector retrieval is not needed unless the LLM planner asks for it.",
        }
    if route == "hook":
        return {
            "route": route,
            "needs_structured_metrics": False,
            "balanced_retrieval": True,
            "doc_types": ["hook_0_5s", "hook_0_10s", "creative_features"],
            "per_video": 3,
            "use_comment_fact_tool": False,
            "tool_only": False,
            "reason": "Hook comparison needs balanced A/B hook and creative feature evidence.",
        }
    if route == "comments":
        requested_video = requested_video_id(question)
        needs_comment_facts = asks_for_comment_fact_tool(question, history or [])
        return {
            "route": route,
            "needs_structured_metrics": False,
            "balanced_retrieval": requested_video is None,
            "video_id": requested_video,
            "doc_types": [
                "top_comments",
                "comment_intelligence_summary",
                "comment_theme",
                "comment_cluster",
                "comment_noise_summary",
            ],
            "per_video": 5,
            "n_results": 10,
            "use_comment_fact_tool": needs_comment_facts,
            "tool_only": needs_comment_facts,
            "reason": "Comment questions need raw top-comment groups plus processed comment intelligence.",
        }
    if route == "improvement":
        return {
            "route": route,
            "needs_structured_metrics": True,
            "balanced_retrieval": True,
            "doc_types": [
                "creative_features",
                "comment_intelligence_summary",
                "comment_theme",
                "hook_0_5s",
                "hook_0_10s",
                "transcript_window",
                "transcript_text_window",
            ],
            "per_video": 6,
            "use_comment_fact_tool": False,
            "tool_only": False,
            "reason": "Recommendations need balanced creative, transcript, and audience-reaction evidence.",
        }
    if route == "comparison":
        return {
            "route": route,
            "needs_structured_metrics": True,
            "balanced_retrieval": True,
            "doc_types": [
                "creative_features",
                "comment_intelligence_summary",
                "comment_theme",
                "comment_cluster",
                "hook_0_5s",
                "hook_0_10s",
                "transcript_window",
                "transcript_text_window",
            ],
            "per_video": 6,
            "use_comment_fact_tool": False,
            "tool_only": False,
            "reason": "Comparison needs balanced A/B performance, content, and audience signals.",
        }
    return {
        "route": route,
        "needs_structured_metrics": True,
        "balanced_retrieval": False,
        "doc_types": [
            "full_transcript",
            "transcript_window",
            "transcript_text_window",
            "creative_features",
            "comment_intelligence_summary",
            "comment_theme",
            "comment_cluster",
            "video_fact_card",
        ],
        "n_results": 8,
        "use_comment_fact_tool": False,
        "tool_only": False,
        "reason": "General answer uses broad but budgeted retrieval.",
    }


def requested_video_id(question: str, history: list[Message] | None = None) -> str | None:
    question = question.lower()
    if re.search(r"\bvideo\s+a\b", question):
        return "A"
    if re.search(r"\bvideo\s+b\b", question):
        return "B"
    if should_resolve_from_history(question):
        return last_referenced_video_id(history or [])
    return None


def should_resolve_from_history(question: str) -> bool:
    text = question.lower()
    if any(term in text for term in ["each", "both", "compare", "video a and video b", "a vs b"]):
        return False
    return bool(
        re.search(r"\b(it|its|they|their|that|this|same|previous|one)\b", text)
        or text.strip().startswith(("and ", "what about", "how about"))
    )


def last_referenced_video_id(history: list[Message]) -> str | None:
    for message in reversed(history):
        content = str(message.get("content") or "").lower()
        matches = re.findall(r"\bvideo\s+([ab])\b", content)
        if matches:
            return matches[-1].upper()
    return None
