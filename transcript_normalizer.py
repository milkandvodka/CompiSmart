from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

import requests

from comparag.config import get_gemini_api_key


SCRIPT_RANGES = {
    "devanagari": r"[\u0900-\u097F]",
    "bengali": r"[\u0980-\u09FF]",
    "gurmukhi": r"[\u0A00-\u0A7F]",
    "gujarati": r"[\u0A80-\u0AFF]",
    "oriya": r"[\u0B00-\u0B7F]",
    "tamil": r"[\u0B80-\u0BFF]",
    "telugu": r"[\u0C00-\u0C7F]",
    "kannada": r"[\u0C80-\u0CFF]",
    "malayalam": r"[\u0D00-\u0D7F]",
    "sinhala": r"[\u0D80-\u0DFF]",
    "arabic_urdu": r"[\u0600-\u06FF]",
    "thai": r"[\u0E00-\u0E7F]",
    "cjk": r"[\u4E00-\u9FFF]",
}


def detect_scripts(text: str) -> list[str]:
    return [name for name, pattern in SCRIPT_RANGES.items() if re.search(pattern, text)]


def needs_llm_normalization(text: str) -> bool:
    return bool(detect_scripts(text))


def transcript_language_needs_normalization(transcript: dict[str, Any]) -> bool:
    language = str(transcript.get("language") or "").strip().lower()
    if not language:
        return False
    return language not in {"en", "eng", "english"}


def normalize_with_gemini(
    text: str,
    *,
    api_key: str,
    model: str,
    timeout_seconds: float,
    video_context: str = "",
) -> dict[str, Any]:
    prompt = build_normalization_prompt(text, video_context=video_context)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    response = requests.post(
        url,
        headers={"x-goog-api-key": api_key},
        json={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    text_response = payload["candidates"][0]["content"]["parts"][0]["text"]
    return parse_json_response(text_response)


def normalize_with_completion(
    text: str,
    *,
    complete: Callable[[str], str],
    video_context: str = "",
) -> dict[str, Any]:
    return parse_json_response(complete(build_normalization_prompt(text, video_context=video_context)))


def build_normalization_prompt(text: str, *, video_context: str = "") -> str:
    context_block = f"\nVideo metadata context:\n{video_context}\n" if video_context else ""
    return f"""
You convert multilingual social-video transcripts into retrieval-friendly text.

Return strict JSON only with these keys:
- detected_language_labels: array of concise labels, e.g. ["Hindi", "English", "Hinglish"]
- hinglish_latin: Latin-script Hinglish transliteration that preserves spoken style and English words
- english_normalized: clean English translation/normalization for semantic search and RAG
- notes: short array of uncertainty notes, empty when none

Rules:
- Preserve brand names, product names, hashtags, numbers, dates, claims, ingredients, and measurements.
- Do not add facts that are not present.
- Do not summarize away details.
- Keep english_normalized natural and query-friendly.
- Keep hinglish_latin in English alphabet only.
- Use video metadata only to resolve likely ASR-garbled proper nouns, route names, hashtags, or number words.
- If a place name or distance is uncertain, preserve the likely reading and mention the uncertainty in notes instead of deleting it.
{context_block}

Transcript:
{text}
""".strip()


def parse_json_response(text_response: str) -> dict[str, Any]:
    stripped = text_response.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def english_only_variant(text: str) -> dict[str, Any]:
    return {
        "detected_language_labels": ["English"],
        "hinglish_latin": text,
        "english_normalized": text,
        "notes": [],
    }


def normalize_result_payload(
    payload: dict[str, Any],
    *,
    api_key: str | None,
    model: str,
    timeout_seconds: float,
    complete: Callable[[str], str] | None = None,
    provider_label: str | None = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    for video in payload.get("videos", []):
        transcript = video.get("transcript") or {}
        if skip_existing and (transcript.get("variants") or {}).get("english_normalized"):
            continue
        text = transcript.get("text") or ""
        video_context = format_video_context(video)
        scripts = detect_scripts(text)
        language_needs_normalization = transcript_language_needs_normalization(transcript)
        variants = {
            "original": text,
            "detected_scripts": scripts,
            "source_language": transcript.get("language"),
            "llm_used": False,
            "model": None,
            "detected_language_labels": [],
            "hinglish_latin": text,
            "english_normalized": text,
            "notes": [],
        }

        if text and (needs_llm_normalization(text) or language_needs_normalization):
            if complete:
                normalized = normalize_with_completion(text, complete=complete, video_context=video_context)
                variants.update(normalized)
                variants["llm_used"] = True
                variants["model"] = provider_label or model
            elif not api_key:
                if scripts:
                    variants["notes"] = ["Non-English script detected but GEMINI_API_KEY was not provided."]
                else:
                    variants["notes"] = [
                        f"Transcript language '{transcript.get('language')}' needs normalization but GEMINI_API_KEY was not provided."
                    ]
            else:
                normalized = normalize_with_gemini(
                    text,
                    api_key=api_key,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    video_context=video_context,
                )
                variants.update(normalized)
                variants["llm_used"] = True
                variants["model"] = model
        elif text:
            variants.update(english_only_variant(text))

        transcript["variants"] = variants
        video["transcript"] = transcript
    return payload


def format_video_context(video: dict[str, Any]) -> str:
    fields: list[str] = []
    for key in ("title", "description", "creator", "upload_date"):
        value = video.get(key)
        if value:
            fields.append(f"{key}: {value}")
    hashtags = video.get("hashtags") or []
    if hashtags:
        if isinstance(hashtags, list):
            fields.append("hashtags: " + ", ".join(str(tag) for tag in hashtags[:20]))
        else:
            fields.append(f"hashtags: {hashtags}")
    return "\n".join(fields)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create RAG-friendly transcript variants.")
    parser.add_argument("--input", "-i", required=True, help="Input extractor JSON.")
    parser.add_argument("--output", "-o", required=True, help="Output JSON with transcript variants.")
    parser.add_argument("--model", default="gemini-2.5-flash-lite", help="Gemini model for normalization.")
    parser.add_argument(
        "--api-key",
        help="Gemini API key. Prefer GEMINI_API_KEY, GOOGLE_API_KEY, GOOGLE_GENAI_API_KEY, or local .env.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=45, help="Gemini request timeout.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    api_key = get_gemini_api_key(args.api_key)
    input_path = Path(args.input)
    output_path = Path(args.output)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    normalized = normalize_result_payload(
        payload,
        api_key=api_key,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )
    output_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
