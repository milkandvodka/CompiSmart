from __future__ import annotations

import re


def phrase_tasks(
    phrases: list[str],
    requested_video: str | None,
    *,
    map_unscoped_phrases: bool = False,
) -> list[dict[str, str | None]]:
    if requested_video or len(phrases) == 1:
        return [{"phrase": phrase, "video_id": requested_video} for phrase in phrases]
    if map_unscoped_phrases:
        video_ids = ["B", "A"]
        return [
            {"phrase": phrase, "video_id": video_ids[index] if index < len(video_ids) else None}
            for index, phrase in enumerate(phrases)
        ]
    return [{"phrase": phrase, "video_id": None} for phrase in phrases]


def extract_comment_phrases(question: str) -> list[str]:
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", question)
    if quoted:
        return [phrase.strip() for phrase in quoted if phrase.strip()]
    keyword_terms = extract_keyword_terms(question)
    if keyword_terms:
        return keyword_terms
    matches = re.findall(
        r"(?:commented|wrote|said)\s+(?:something\s+)?(?:about\s+)?([a-zA-Z0-9][a-zA-Z0-9 ]{1,40})(?=\s+(?:in|on|and|for|from)\b|[?.!,]|$)",
        question,
        flags=re.I,
    )
    return [clean_comment_phrase(match) for match in matches if clean_comment_phrase(match)]


def extract_keyword_terms(question: str) -> list[str]:
    matches = re.findall(
        r"(?:about|mention(?:ed|ing|s)?|containing|contains)\s+([^?.]+)",
        question,
        flags=re.I,
    )
    terms: list[str] = []
    for match in matches:
        normalized = re.sub(r"\b(?:or|and)\b", ",", match, flags=re.I).replace("/", ",")
        for raw in normalized.split(","):
            phrase = clean_comment_phrase(raw)
            if phrase and phrase.lower() not in {item.lower() for item in terms}:
                terms.append(phrase)
    return terms


def clean_comment_phrase(value: str) -> str:
    phrase = value.strip(" \t\r\n'\".,:;!?")
    phrase = re.sub(
        r"^(?:the\s+)?(?:same\s+)?(?:word|words|phrase|phrases|term|terms|comment|comments)\s+",
        "",
        phrase,
        flags=re.I,
    ).strip(" \t\r\n'\".,:;!?")
    if phrase.lower() in {"something", "anything", "comment", "comments", "same"}:
        return ""
    return phrase


def should_map_unscoped_phrases_to_platforms(question: str, phrases: list[str]) -> bool:
    if len(phrases) < 2:
        return False
    if not re.search(r"['\"]", question):
        return False
    text = question.lower()
    return bool(("insta" in text or "instagram" in text) and ("yt" in text or "youtube" in text))


def should_resolve_comment_phrase_from_history(text: str) -> bool:
    return any(term in text for term in ["their", "they", "them", "again", "same", "those"]) and any(
        term in text for term in ["like", "likes", "comment", "comments", "user", "profile"]
    )


def extract_comment_phrases_from_history(history: list[dict[str, str]], max_phrases: int = 2) -> list[str]:
    phrases: list[str] = []
    for message in reversed(history[-6:]):
        content = str(message.get("content") or "")
        for pattern in [
            r"Phrase ['\"]([^'\"]+)['\"]",
            r"phrase \*\*['\"]?([^'\"*]+)['\"]?\*\*",
            r"commenters?\s+wrote\s+[\"'\u201c\u201d]([^\"'\u201c\u201d]+)[\"'\u201c\u201d]",
            r"comments?\s+(?:saying|said|with)\s+[\"'\u201c\u201d]([^\"'\u201c\u201d]+)[\"'\u201c\u201d]",
            r"(?:matching|matched|matches)\s+[\"'\u201c\u201d]([^\"'\u201c\u201d]+)[\"'\u201c\u201d]",
        ]:
            for phrase in re.findall(pattern, content, flags=re.I):
                cleaned = phrase.strip(" ,.:;!?")
                if cleaned and cleaned.lower() not in {item.lower() for item in phrases}:
                    phrases.append(cleaned)
                if len(phrases) >= max_phrases:
                    return phrases
    return phrases


def extract_comment_ids_from_history(history: list[dict[str, str]], max_ids: int = 12) -> list[str]:
    comment_ids: list[str] = []
    for message in reversed(history[-6:]):
        content = str(message.get("content") or "")
        for comment_id in re.findall(r"\[?Video\s+[AB],\s+comment\s+([A-Za-z0-9_.:-]+)\]?", content):
            if comment_id not in comment_ids:
                comment_ids.append(comment_id)
            if len(comment_ids) >= max_ids:
                return comment_ids
    return comment_ids


def requested_video_from_text(text: str) -> str | None:
    mentions_a = bool(re.search(r"\b(video\s+a|youtube|yt)\b", text))
    mentions_b = bool(re.search(r"\b(video\s+b|instagram|insta|ig)\b", text))
    if mentions_a and mentions_b:
        return None
    if mentions_a:
        return "A"
    if mentions_b:
        return "B"
    return None


def asks_most_liked_commenter(text: str) -> bool:
    return any(
        term in text
        for term in [
            "most likes",
            "most liked",
            "most comment likes",
            "highest likes",
            "highest comment likes",
            "top commenter",
        ]
    ) and any(
        term in text for term in ["user", "profile", "commenter", "who"]
    )


def asks_for_single_top_user(text: str) -> bool:
    return "the user" in text or "user who had the most" in text or "profile of the user" in text


def asks_top_comments(text: str) -> bool:
    return (
        ("top" in text and "comment" in text)
        or "most liked comment" in text
        or "fetched comment" in text
        or "every comment" in text
        or "all comments" in text
        or "list comments" in text
    )
