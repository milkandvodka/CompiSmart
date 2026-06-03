from __future__ import annotations

from typing import Any

from ..context import DEFAULT_CONTEXT_PROFILE, ModelContextProfile
from ..memory import Message
from ..metrics import format_metric
from .state import ChatState
from .utils import fit_to_budget


def build_prompt(state: ChatState, citations: list[dict[str, Any]]) -> str:
    route = state.get("route", "general")
    profile_block = format_profiles(state.get("profiles", []))
    context_profile = state.get("context_profile", DEFAULT_CONTEXT_PROFILE)
    history_block = format_memory(
        state.get("history", []),
        memory_summary=state.get("memory_summary") or "",
        context_profile=context_profile,
    )
    tool_block = format_tool_results(state.get("tool_results", {}), context_profile)
    context_block = format_retrieved(state.get("retrieved", []), context_profile)
    evidence_plan_block = format_evidence_plan(state.get("evidence_plan", {}))
    citation_block = "\n".join(f"- {item['label']}" for item in citations) or "- No retrieved source chunks."
    prompt = f"""
You are an expert creator analytics assistant comparing exactly two social videos.

Rules:
- Answer the creator's question directly.
- Use only the structured metrics, exact tool results, retrieved context, and conversation memory in this prompt. Do not use outside/world knowledge, map estimates, or generic textbook explanations.
- If the user asks multiple things, answer each part in its own short section and do not drop subquestions.
- Use structured metrics for exact numbers.
- Engagement rate always means (likes + comments) / views * 100. Never use follower count as the denominator.
- Use retrieved transcript/comment chunks for content claims.
- For algorithm, hook, creative, and content questions, summarize the retrieved video evidence; do not replace it with a generic explanation.
- Every factual answer section must be backed by an available citation label. If you cannot cite the claim, do not include it.
- Cite claims with bracketed labels such as [Video A, transcript 00:00-00:05].
- Put only one citation label inside each bracket. Use [Video A, source] [Video B, source], not one combined bracket.
- Cite exact metrics and engagement-rate claims with metadata snapshot labels.
- If the evidence is approximate, say so briefly.
- Do not invent unavailable metrics.
- Do not mark a requested fact unavailable until you have checked retrieved transcript, comment, and metadata evidence.
- If retrieved evidence is relevant but ASR/normalization looks ambiguous, say what the evidence says and clearly note the ambiguity.
- Never answer a compound question with one blanket "not enough context" if any subpart has relevant evidence. Split the answer into available evidence and missing/ambiguous evidence.
- If a requested name, place, route, or phrase is not present exactly, but retrieved evidence contains a close or possibly confused term, say "I do not see <requested term>; the closest retrieved evidence says <source wording>" and cite it.
- If a requested event/action is absent but related transcript evidence is present, answer the related transcript evidence first, then say which exact event/action is not in the retrieved evidence.
- If there is not enough relevant indexed evidence to answer a subquestion accurately, say "not enough context" only for that subquestion and name the missing evidence instead of guessing.
- Preserve wording from evidence. Do not relabel, merge, or reinterpret separate facts into a cleaner answer than the source supports.
- Do not dump citation labels as standalone lines; cite inline only where they support a claim.
- Keep advice specific and actionable.
- Avoid generic report-style headings unless the user asks for a report.

Route: {route}

Evidence plan:
{evidence_plan_block}

Conversation memory:
{history_block}

Structured video metrics:
{profile_block}

Exact tool results:
{tool_block}

Retrieved context:
{context_block}

Available citation labels:
{citation_block}

User question:
{state["question"]}

Final grounding checklist:
- Answer only with supplied evidence and available citation labels.
- Before saying "not enough context", check retrieved transcript, comments, metadata, exact tool results, and memory.
- If evidence is partial, answer the supported part with citations and mark only the unsupported part as "not enough context".
- If evidence is ambiguous or the user's wording may be a typo/ASR mismatch, say what the evidence actually says and cite it; do not silently treat the mismatch as no evidence.
- Do not include map estimates, textbook explanations, or uncited factual claims.
""".strip()
    return fit_to_budget(prompt, context_profile.max_prompt_chars)


def generation_failure_answer(state: dict[str, Any], error: str) -> str:
    citations = "\n".join(f"- {item.get('label')}" for item in state.get("citations", [])[:12]) or "- none"
    return (
        "The LLM generation call failed, so I am not going to fabricate an answer from heuristics.\n\n"
        f"Error: {error}\n\n"
        "Evidence that had been prepared for the model:\n"
        f"Route: {state.get('route')}\n"
        f"Evidence plan: {state.get('evidence_plan')}\n"
        f"Citations available:\n{citations}"
    )


def format_profiles(profiles: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for profile in profiles:
        engagement = profile.get("engagement_rate")
        engagement_text = "unknown" if engagement is None else f"{float(engagement):.2f}%"
        hashtags = profile.get("hashtags") or []
        if isinstance(hashtags, list):
            hashtags_text = ", ".join(str(tag) for tag in hashtags[:12]) or "none"
        else:
            hashtags_text = str(hashtags)
        lines.append(
            "\n".join(
                [
                    f"Video {profile.get('video_id')}:",
                    f"  platform: {profile.get('platform')}",
                    f"  title: {profile.get('title') or 'unknown'}",
                    f"  creator: {profile.get('creator') or 'unknown'}",
                    f"  follower_count: {format_metric(profile.get('follower_count'))}",
                    f"  views: {format_metric(profile.get('views'))}",
                    f"  likes: {format_metric(profile.get('likes'))}",
                    f"  comments: {format_metric(profile.get('comments'))}",
                    "  engagement_rate_formula: (likes + comments) / views * 100",
                    f"  engagement_rate: {engagement_text}",
                    f"  upload_date: {profile.get('upload_date') or 'unknown'}",
                    f"  duration_seconds: {format_metric(profile.get('duration_seconds'))}",
                    f"  hashtags: {hashtags_text}",
                    f"  thumbnail: {profile.get('thumbnail') or 'unavailable'}",
                    f"  creator_url: {profile.get('creator_url') or 'unavailable'}",
                ]
            )
        )
    return "\n\n".join(lines)


def format_history(
    history: list[Message],
    max_messages: int = 8,
    context_profile: ModelContextProfile = DEFAULT_CONTEXT_PROFILE,
) -> str:
    if not history:
        return "No prior turns."
    lines = [f"{message.get('role')}: {message.get('content')}" for message in history[-max_messages:]]
    return fit_to_budget("\n".join(lines), context_profile.max_history_chars)


def format_memory(
    history: list[Message],
    *,
    memory_summary: str = "",
    max_messages: int = 8,
    context_profile: ModelContextProfile = DEFAULT_CONTEXT_PROFILE,
) -> str:
    blocks = []
    if memory_summary:
        blocks.append(
            "Long-term summary:\n"
            + fit_to_budget(memory_summary, context_profile.max_memory_summary_chars)
        )
    blocks.append("Recent turns:\n" + format_history(history, max_messages=max_messages, context_profile=context_profile))
    return "\n\n".join(blocks)


def build_memory_summary_prompt(*, existing_summary: str, recent_messages: list[Message], max_chars: int) -> str:
    recent_block = "\n".join(f"{message.get('role')}: {message.get('content')}" for message in recent_messages)
    return f"""
You are maintaining long-term memory for a creator analytics RAG chatbot.

Update the memory summary using the existing summary and recent turns. Return plain text only.

Rules:
- Preserve durable user preferences, decisions, unresolved tasks, and references needed for follow-up questions.
- Preserve exact Video A/B references and exact numeric facts only when they appeared in the conversation.
- Do not infer new facts, strategies, sentiment, or conclusions.
- Do not include API keys, passwords, JWTs, access tokens, service-role keys, or other secrets.
- Do not copy huge signed media URLs or large metadata tables into memory; note that thumbnail/profile URLs are available from structured data instead.
- Remove duplicated tool output and transient errors unless the user needs to remember them.
- Keep it under {max_chars} characters.

Existing summary:
{existing_summary or "[none]"}

Recent turns:
{recent_block or "[none]"}
""".strip()


def format_evidence_plan(evidence_plan: dict[str, Any]) -> str:
    if not evidence_plan:
        return "No evidence plan."
    return "\n".join(
        [
            f"planner_mode: {evidence_plan.get('planner_mode')}",
            f"needs_structured_metrics: {evidence_plan.get('needs_structured_metrics')}",
            f"balanced_retrieval: {evidence_plan.get('balanced_retrieval')}",
            f"use_comment_fact_tool: {evidence_plan.get('use_comment_fact_tool')}",
            f"doc_types: {evidence_plan.get('doc_types') or []}",
            f"reason: {evidence_plan.get('reason')}",
        ]
    )


def format_tool_results(
    tool_results: dict[str, Any],
    context_profile: ModelContextProfile = DEFAULT_CONTEXT_PROFILE,
) -> str:
    if not tool_results:
        return "No exact tools were called."
    if not tool_results.get("available"):
        return f"No exact tool result. Note: {tool_results.get('note') or 'unavailable'}"
    return fit_to_budget(str(tool_results.get("answer_text") or ""), context_profile.max_tool_result_chars)


def format_retrieved(
    chunks: list[dict[str, Any]],
    context_profile: ModelContextProfile = DEFAULT_CONTEXT_PROFILE,
) -> str:
    if not chunks:
        return "No vector chunks retrieved for this route."
    blocks = []
    used_chars = 0
    for chunk in chunks:
        label = chunk.get("metadata", {}).get("citation_label") or chunk.get("id")
        text = fit_to_budget(str(chunk.get("text") or ""), context_profile.max_retrieved_chunk_chars)
        text = append_transcript_variants(text, chunk, context_profile)
        block = f"[{label}]\n{text}"
        if used_chars + len(block) > context_profile.max_retrieved_context_chars:
            remaining = context_profile.max_retrieved_context_chars - used_chars
            if remaining > 200:
                blocks.append(fit_to_budget(block, remaining))
            break
        blocks.append(block)
        used_chars += len(block)
    return "\n\n".join(blocks)


def append_transcript_variants(text: str, chunk: dict[str, Any], context_profile: ModelContextProfile) -> str:
    metadata = chunk.get("metadata") or {}
    doc_type = str(metadata.get("doc_type") or chunk.get("doc_type") or "")
    chunk_id = str(chunk.get("id") or "")
    is_transcript_chunk = "transcript" in doc_type or "_transcript_" in chunk_id
    if not is_transcript_chunk:
        return text
    additions = []
    hinglish = str(metadata.get("hinglish_text") or "").strip()
    raw = str(metadata.get("raw_text") or "").strip()
    if hinglish and hinglish not in text:
        additions.append("Hinglish/raw-latin variant: " + fit_to_budget(hinglish, 600))
    if raw and raw not in text:
        additions.append("Original ASR transcript: " + fit_to_budget(raw, 600))
    if not additions:
        return text
    return fit_to_budget(text + "\n" + "\n".join(additions), context_profile.max_retrieved_chunk_chars + 1400)
