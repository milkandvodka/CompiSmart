from __future__ import annotations

STRUCTURED_ROUTES = {"metrics", "creator"}
ALLOWED_ROUTES = {"metrics", "creator", "comments", "hook", "improvement", "comparison", "general"}
ALLOWED_DOC_TYPES = {
    "full_transcript",
    "transcript_window",
    "transcript_text_window",
    "hook_0_5s",
    "hook_0_10s",
    "creative_features",
    "comment_intelligence_summary",
    "comment_theme",
    "comment_cluster",
    "comment_noise_summary",
    "top_comments",
    "video_fact_card",
}
MAX_HISTORY_CHARS = 3000
MAX_RETRIEVED_CONTEXT_CHARS = 9000
MAX_RETRIEVED_CHUNK_CHARS = 1800
