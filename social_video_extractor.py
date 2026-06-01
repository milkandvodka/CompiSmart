from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import html
import json
import os
import re
import time as time_module
import urllib.parse
import wave
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}

INSTAGRAM_HOSTS = {
    "instagram.com",
    "www.instagram.com",
    "m.instagram.com",
}


class ExtractorError(RuntimeError):
    pass


class URLValidationError(ValueError):
    pass


@dataclass
class TranscriptSegment:
    start: float | None
    duration: float | None
    text: str
    words: list[dict[str, Any]] | None = None


@dataclass
class TranscriptResult:
    available: bool
    text: str
    segments: list[TranscriptSegment]
    language: str | None
    source: str | None
    kind: str | None
    note: str | None = None
    engine: str | None = None
    model: str | None = None
    language_probability: float | None = None
    audio_path: str | None = None


def detect_platform(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower().split("@")[-1].split(":")[0]
    path = parsed.path.lower()

    if host in YOUTUBE_HOSTS:
        return "youtube"
    if host in INSTAGRAM_HOSTS and re.search(r"/reels?/", path):
        return "instagram_reel"
    if host in INSTAGRAM_HOSTS and re.search(r"/p/", path):
        return "instagram_post"
    return None


def is_instagram_media(platform: str | None) -> bool:
    return platform in {"instagram_reel", "instagram_post"}


def extract_youtube_video_id(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower().split(":")[0]
    path_parts = [part for part in parsed.path.split("/") if part]

    if host in {"youtu.be", "www.youtu.be"} and path_parts:
        return path_parts[0]

    query = urllib.parse.parse_qs(parsed.query)
    if query.get("v"):
        return query["v"][0]

    if len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts", "live"}:
        return path_parts[1]

    return None


def resolve_input_urls(args: argparse.Namespace) -> tuple[str, str]:
    youtube_url = args.youtube_url
    instagram_url = args.instagram_url
    positional_urls = args.urls or []

    if youtube_url or instagram_url:
        if positional_urls:
            raise URLValidationError("Use either named URL options or positional URLs, not both.")
        if not youtube_url or not instagram_url:
            raise URLValidationError("Both --youtube-url and --instagram-url are mandatory.")
    else:
        if len(positional_urls) != 2:
            raise URLValidationError("Pass exactly two URLs: one YouTube URL and one Instagram media URL.")

        platforms = {detect_platform(url): url for url in positional_urls}
        youtube_url = platforms.get("youtube")
        instagram_url = next((url for platform, url in platforms.items() if is_instagram_media(platform)), None)
        if not youtube_url or not instagram_url:
            raise URLValidationError("The two URLs must include one YouTube URL and one Instagram media URL.")

    if detect_platform(youtube_url) != "youtube":
        raise URLValidationError("--youtube-url must be a valid YouTube video URL.")
    if not is_instagram_media(detect_platform(instagram_url)):
        raise URLValidationError("--instagram-url must be a valid Instagram Reel or post URL.")

    return youtube_url, instagram_url


def load_yt_dlp() -> Any:
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:
        raise ExtractorError(
            "Missing dependency: yt-dlp. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return yt_dlp


def load_faster_whisper() -> Any:
    try:
        import faster_whisper  # type: ignore
    except ImportError as exc:
        raise ExtractorError(
            "Missing dependency: faster-whisper. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return faster_whisper


def load_instaloader() -> Any:
    try:
        import instaloader  # type: ignore
    except ImportError as exc:
        raise ExtractorError(
            "Missing dependency: instaloader. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return instaloader


def load_instagrapi() -> Any:
    try:
        import instagrapi  # type: ignore
    except ImportError as exc:
        raise ExtractorError(
            "Missing dependency: instagrapi. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return instagrapi


def load_huggingface_hub() -> Any:
    try:
        import huggingface_hub  # type: ignore
    except ImportError as exc:
        raise ExtractorError(
            "Missing dependency: huggingface-hub. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return huggingface_hub


def extract_info(
    url: str,
    *,
    cookies: str | None,
    cookies_from_browser: str | None,
    fetch_comments: bool,
) -> dict[str, Any]:
    yt_dlp = load_yt_dlp()
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["all"],
        "socket_timeout": 30,
    }

    if cookies:
        opts["cookiefile"] = cookies
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if fetch_comments:
        opts["getcomments"] = True

    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def download_audio_for_asr(
    url: str,
    *,
    platform: str,
    video_id: str | None,
    cache_dir: Path,
    cookies: str | None,
    cookies_from_browser: str | None,
) -> Path:
    yt_dlp = load_yt_dlp()
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_id or "media").strip("._") or "media"
    output_template = str(cache_dir / f"{platform}_{safe_id}.%(ext)s")

    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "socket_timeout": 30,
    }
    if cookies:
        opts["cookiefile"] = cookies
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        candidates = downloaded_file_candidates(info, ydl)

    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path

    globbed = sorted(cache_dir.glob(f"{platform}_{safe_id}.*"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in globbed:
        if path.is_file():
            return path

    raise ExtractorError(f"Audio download completed, but no local media file was found for {url}")


def downloaded_file_candidates(info: dict[str, Any], ydl: Any) -> list[str]:
    candidates: list[str] = []
    for download in info.get("requested_downloads") or []:
        if isinstance(download, dict):
            for key in ["filepath", "filename", "__finaldir"]:
                value = download.get(key)
                if isinstance(value, str):
                    candidates.append(value)

    for key in ["filepath", "_filename", "filename"]:
        value = info.get(key)
        if isinstance(value, str):
            candidates.append(value)

    try:
        candidates.append(ydl.prepare_filename(info))
    except Exception:
        pass

    return candidates


def first_present(info: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = info.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def iso_upload_date(info: dict[str, Any]) -> str | None:
    timestamp = info.get("timestamp") or info.get("release_timestamp") or info.get("modified_timestamp")
    if timestamp:
        try:
            return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).date().isoformat()
        except (TypeError, ValueError, OSError):
            pass

    upload_date = info.get("upload_date") or info.get("release_date")
    if isinstance(upload_date, str) and re.fullmatch(r"\d{8}", upload_date):
        return f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"

    return None


def collect_hashtags(info: dict[str, Any]) -> list[str]:
    tags = info.get("tags") or []
    searchable_text = " ".join(
        value
        for value in [info.get("title"), info.get("fulltitle"), info.get("description")]
        if isinstance(value, str)
    )
    hashtags: set[str] = set()

    for tag in tags:
        if not isinstance(tag, str):
            continue
        clean_tag = tag.strip().lstrip("#")
        if clean_tag:
            hashtags.add(clean_tag)

    for match in re.findall(r"(?<!\w)#([\w.]+)", searchable_text, flags=re.UNICODE):
        hashtags.add(match.strip("."))

    return sorted(hashtags, key=str.lower)


def choose_caption_track(
    info: dict[str, Any],
    preferred_language: str,
) -> tuple[str, str, dict[str, Any]] | None:
    caption_groups = [
        ("manual", info.get("subtitles") or {}),
        ("automatic", info.get("automatic_captions") or {}),
    ]

    for kind, captions in caption_groups:
        if not isinstance(captions, dict) or not captions:
            continue

        language = choose_language(captions, preferred_language)
        if not language:
            continue

        entries = captions.get(language) or []
        if not isinstance(entries, list):
            continue

        entry = choose_caption_entry(entries)
        if entry:
            return kind, language, entry

    return None


def choose_language(captions: dict[str, Any], preferred_language: str) -> str | None:
    languages = list(captions)
    if preferred_language in captions:
        return preferred_language

    normalized = preferred_language.lower()
    for language in languages:
        if language.lower() == normalized or language.lower().startswith(f"{normalized}-"):
            return language

    for language in languages:
        if normalized in language.lower():
            return language

    return languages[0] if languages else None


def choose_caption_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    preferred_exts = ["json3", "vtt", "srt", "ttml", "srv3", "xml"]
    for ext in preferred_exts:
        for entry in entries:
            if entry.get("url") and str(entry.get("ext", "")).lower() == ext:
                return entry
    for entry in entries:
        if entry.get("url"):
            return entry
    return None


def fetch_caption_text(url: str) -> str:
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise ExtractorError(
            "Missing dependency: requests. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def extract_transcript(info: dict[str, Any], preferred_language: str) -> TranscriptResult:
    track = choose_caption_track(info, preferred_language)
    if not track:
        note = "No subtitle or caption track was exposed for this video."
        if is_instagram_media(detect_platform(info.get("webpage_url") or "")):
            note += " Instagram media often needs authenticated cookies and may not publish captions."
        return TranscriptResult(False, "", [], None, None, None, note)

    kind, language, entry = track
    source_url = entry.get("url")
    extension = str(entry.get("ext") or "").lower()

    try:
        caption_text = fetch_caption_text(source_url)
        segments = parse_caption(caption_text, extension)
    except Exception as exc:
        return TranscriptResult(
            False,
            "",
            [],
            language,
            source_url,
            kind,
            f"Caption track was found but could not be read: {exc}",
        )

    text = normalize_transcript_text(" ".join(segment.text for segment in segments))
    return TranscriptResult(
        bool(text),
        text,
        segments,
        language,
        source_url,
        kind,
        None if text else "Caption track was empty.",
    )


def resolve_transcript(
    *,
    info: dict[str, Any],
    url: str,
    platform: str,
    preferred_caption_language: str,
    transcribe_missing: bool,
    asr_provider: str,
    asr_model: str,
    hf_asr_model: str,
    hf_token: str | None,
    asr_timeout_seconds: float,
    asr_language: str | None,
    asr_device: str,
    asr_compute_type: str,
    asr_cache_dir: Path,
    keep_audio: bool,
    cookies: str | None,
    cookies_from_browser: str | None,
) -> TranscriptResult:
    transcript = extract_transcript(info, preferred_caption_language)
    if transcript.available or not transcribe_missing:
        return transcript

    original_note = transcript.note
    audio_path: Path | None = None
    try:
        audio_path = download_audio_for_asr(
            url,
            platform=platform,
            video_id=str(info.get("id") or ""),
            cache_dir=asr_cache_dir,
            cookies=cookies,
            cookies_from_browser=cookies_from_browser,
        )
        return transcribe_audio(
            audio_path,
            provider=asr_provider,
            model_name=asr_model,
            hf_model=hf_asr_model,
            hf_token=hf_token,
            timeout_seconds=asr_timeout_seconds,
            language=asr_language,
            device=asr_device,
            compute_type=asr_compute_type,
            previous_note=original_note,
        )
    except Exception as exc:
        note = f"{original_note or 'No platform transcript available'} ASR fallback failed: {exc}"
        return TranscriptResult(False, "", [], None, None, None, note)
    finally:
        if audio_path and not keep_audio:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass


def transcribe_audio(
    audio_path: Path,
    *,
    provider: str,
    model_name: str,
    hf_model: str,
    hf_token: str | None,
    timeout_seconds: float,
    language: str | None,
    device: str,
    compute_type: str,
    previous_note: str | None,
) -> TranscriptResult:
    selected_provider = choose_asr_provider(provider, hf_token)
    if selected_provider == "hf":
        try:
            return transcribe_audio_with_hugging_face(
                audio_path,
                model_name=hf_model,
                token=hf_token,
                timeout_seconds=timeout_seconds,
                previous_note=previous_note,
            )
        except Exception as exc:
            if provider == "hf":
                raise
            previous_note = f"{previous_note or ''} Hosted ASR failed and local fallback was used: {exc}".strip()

    return transcribe_audio_with_faster_whisper(
        audio_path,
        model_name=model_name,
        language=language,
        device=device,
        compute_type=compute_type,
        previous_note=previous_note,
    )


def choose_asr_provider(provider: str, hf_token: str | None) -> str:
    normalized = provider.lower()
    if normalized not in {"auto", "local", "hf"}:
        raise ExtractorError("--asr-provider must be one of: auto, local, hf")
    if normalized == "auto":
        return "hf" if hf_token else "local"
    return normalized


def transcribe_audio_with_hugging_face(
    audio_path: Path,
    *,
    model_name: str,
    token: str | None,
    timeout_seconds: float,
    previous_note: str | None,
) -> TranscriptResult:
    if not token:
        raise ExtractorError("HF_TOKEN or --hf-token is required for --asr-provider hf.")

    huggingface_hub = load_huggingface_hub()
    wav_path = ensure_wav_for_hf(audio_path)
    client = huggingface_hub.InferenceClient(
        provider="hf-inference",
        token=token,
        timeout=timeout_seconds,
    )
    output = client.automatic_speech_recognition(str(wav_path), model=model_name)
    payload = output.model_dump() if hasattr(output, "model_dump") else output
    text = normalize_transcript_text(payload.get("text", "") if isinstance(payload, dict) else str(payload))
    duration = audio_duration_seconds(wav_path)
    segment = TranscriptSegment(start=0.0, duration=duration, text=text) if text else None
    note = None
    if previous_note:
        note = f"Platform captions were unavailable; transcript generated with hosted ASR. {previous_note}"
    return TranscriptResult(
        bool(text),
        text,
        [segment] if segment else [],
        None,
        str(wav_path),
        "asr",
        note if text else "Hosted ASR ran but did not produce transcript text.",
        engine="huggingface-inference",
        model=model_name,
        language_probability=None,
        audio_path=str(audio_path),
    )


def ensure_wav_for_hf(audio_path: Path) -> Path:
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    wav_path = audio_path.with_suffix(".wav")
    if wav_path.exists() and wav_path.stat().st_mtime >= audio_path.stat().st_mtime:
        return wav_path

    try:
        import av  # type: ignore
    except ImportError as exc:
        raise ExtractorError("PyAV is required to convert media for hosted ASR.") from exc

    container = av.open(str(audio_path))
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        for packet in container.demux(stream):
            for frame in packet.decode():
                for resampled in resampler.resample(frame):
                    wav_file.writeframes(resampled.to_ndarray().tobytes())
        for resampled in resampler.resample(None):
            wav_file.writeframes(resampled.to_ndarray().tobytes())
    return wav_path


def audio_duration_seconds(audio_path: Path) -> float | None:
    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                return round(wav_file.getnframes() / float(wav_file.getframerate()), 3)
        except Exception:
            return None
    return None


def transcribe_audio_with_faster_whisper(
    audio_path: Path,
    *,
    model_name: str,
    language: str | None,
    device: str,
    compute_type: str,
    previous_note: str | None,
) -> TranscriptResult:
    faster_whisper = load_faster_whisper()
    model = faster_whisper.WhisperModel(model_name, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language=language,
        vad_filter=True,
        word_timestamps=True,
    )
    asr_segments = list(segments_iter)
    transcript_segments = [
        TranscriptSegment(
            start=round(segment.start, 3),
            duration=round(segment.end - segment.start, 3),
            text=normalize_transcript_text(segment.text),
            words=normalize_asr_words(getattr(segment, "words", None)),
        )
        for segment in asr_segments
        if normalize_transcript_text(segment.text)
    ]
    text = normalize_transcript_text(" ".join(segment.text for segment in transcript_segments))
    note = None
    if previous_note:
        note = f"Platform captions were unavailable; transcript generated locally with ASR. {previous_note}"
    return TranscriptResult(
        bool(text),
        text,
        transcript_segments,
        getattr(info, "language", None),
        str(audio_path),
        "asr",
        note if text else "ASR ran but did not produce transcript text.",
        engine="faster-whisper",
        model=model_name,
        language_probability=round(float(getattr(info, "language_probability", 0.0)), 4),
        audio_path=str(audio_path),
    )


def normalize_asr_words(words: Iterable[Any] | None) -> list[dict[str, Any]] | None:
    if not words:
        return None

    normalized: list[dict[str, Any]] = []
    for word in words:
        normalized.append(
            {
                "start": round(float(getattr(word, "start", 0.0)), 3),
                "end": round(float(getattr(word, "end", 0.0)), 3),
                "word": str(getattr(word, "word", "")).strip(),
                "probability": round(float(getattr(word, "probability", 0.0)), 4),
            }
        )
    return normalized


def parse_caption(content: str, extension: str) -> list[TranscriptSegment]:
    content = content.lstrip("\ufeff")
    stripped = content.strip()

    if extension == "json3" or stripped.startswith("{"):
        return parse_json3_caption(stripped)
    if extension in {"ttml", "srv3", "xml"} or stripped.startswith("<"):
        return parse_xml_caption(stripped)
    return parse_vtt_or_srt_caption(stripped)


def parse_json3_caption(content: str) -> list[TranscriptSegment]:
    payload = json.loads(content)
    segments: list[TranscriptSegment] = []

    for event in payload.get("events", []):
        pieces = event.get("segs") or []
        text = "".join(piece.get("utf8", "") for piece in pieces)
        text = normalize_transcript_text(text)
        if not text:
            continue

        start_ms = event.get("tStartMs")
        duration_ms = event.get("dDurationMs")
        segments.append(
            TranscriptSegment(
                start=round(start_ms / 1000, 3) if isinstance(start_ms, (int, float)) else None,
                duration=round(duration_ms / 1000, 3) if isinstance(duration_ms, (int, float)) else None,
                text=text,
            )
        )

    return segments


def parse_vtt_or_srt_caption(content: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    lines = content.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line or line.upper().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            i += 1
            continue

        if "-->" not in line and i + 1 < len(lines) and "-->" in lines[i + 1]:
            i += 1
            line = lines[i].strip()

        if "-->" not in line:
            i += 1
            continue

        start_text, end_text = [part.strip() for part in line.split("-->", 1)]
        end_text = end_text.split()[0]
        start = parse_timestamp(start_text)
        end = parse_timestamp(end_text)
        i += 1

        text_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1

        text = normalize_transcript_text(" ".join(text_lines))
        if text:
            duration = round(end - start, 3) if start is not None and end is not None else None
            segments.append(TranscriptSegment(start=start, duration=duration, text=text))

    return segments


def parse_xml_caption(content: str) -> list[TranscriptSegment]:
    root = ET.fromstring(content)
    segments: list[TranscriptSegment] = []

    for element in root.iter():
        if element.tag.split("}")[-1] != "p":
            continue

        text = normalize_transcript_text(" ".join(element.itertext()))
        if not text:
            continue

        start = parse_duration_text(element.attrib.get("begin") or element.attrib.get("t"))
        duration = parse_duration_text(element.attrib.get("dur") or element.attrib.get("d"))
        end = parse_duration_text(element.attrib.get("end"))
        if duration is None and start is not None and end is not None:
            duration = round(end - start, 3)

        segments.append(TranscriptSegment(start=start, duration=duration, text=text))

    return segments


def parse_timestamp(value: str) -> float | None:
    match = re.search(r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?", value)
    if not match:
        return None

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    millis = int((match.group(4) or "0").ljust(3, "0"))
    return round(hours * 3600 + minutes * 60 + seconds + millis / 1000, 3)


def parse_duration_text(value: str | None) -> float | None:
    if not value:
        return None

    value = value.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return round(float(value) / 1000, 3)
    if value.endswith("ms"):
        return round(float(value[:-2]) / 1000, 3)
    if value.endswith("s"):
        return round(float(value[:-1]), 3)
    return parse_timestamp(value)


def normalize_transcript_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", "", value)
    value = value.replace("\n", " ").replace("\r", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_instagram_shortcode(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) >= 2 and path_parts[0].lower() in {"p", "reel", "reels", "tv"}:
        return path_parts[1]
    return None


def fetch_instagram_supplement(
    url: str,
    *,
    fetch_comments: bool,
    max_comments: int,
    session_user: str | None,
) -> dict[str, Any]:
    shortcode = extract_instagram_shortcode(url)
    if not shortcode:
        return {"available": False, "error": "Could not parse Instagram shortcode from URL."}

    try:
        instaloader = load_instaloader()
        loader = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            save_metadata=False,
            compress_json=False,
            quiet=True,
        )
        if session_user:
            loader.load_session_from_file(session_user)

        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        owner_profile = safe_getattr(post, "owner_profile")
        supplement: dict[str, Any] = {
            "available": True,
            "source": "instaloader",
            "shortcode": safe_getattr(post, "shortcode"),
            "mediaid": safe_getattr(post, "mediaid"),
            "owner_id": safe_getattr(post, "owner_id"),
            "owner_username": safe_getattr(post, "owner_username"),
            "title": safe_getattr(post, "title"),
            "caption": safe_getattr(post, "caption"),
            "caption_hashtags": list(safe_getattr(post, "caption_hashtags") or []),
            "caption_mentions": list(safe_getattr(post, "caption_mentions") or []),
            "tagged_users": list(safe_getattr(post, "tagged_users") or []),
            "likes": safe_getattr(post, "likes"),
            "comments": safe_getattr(post, "comments"),
            "date_utc": isoformat_or_none(safe_getattr(post, "date_utc")),
            "typename": safe_getattr(post, "typename"),
            "is_video": safe_getattr(post, "is_video"),
            "video_duration": safe_getattr(post, "video_duration"),
            "video_view_count": safe_getattr(post, "video_view_count"),
            "video_url": safe_getattr(post, "video_url"),
            "display_url": safe_getattr(post, "url"),
            "accessibility_caption": safe_getattr(post, "accessibility_caption"),
            "location": normalize_instaloader_location(safe_getattr(post, "location")),
            "owner_profile": normalize_instaloader_profile(owner_profile),
            "sidecar_nodes": normalize_instaloader_sidecar(post),
        }
        if fetch_comments:
            supplement["comment_objects"] = normalize_instaloader_comments(post, max_comments)
        return make_json_safe(supplement)
    except Exception as exc:
        return {"available": False, "source": "instaloader", "error": str(exc)}


def fetch_instagram_instagrapi_supplement(
    url: str,
    *,
    media_id: str | None,
    fetch_comments: bool,
    fetch_comment_replies: bool,
    max_comments: int,
    max_comment_replies: int,
    comment_time_budget_seconds: float | None,
    settings_path: Path | None,
    username: str | None,
    password: str | None,
    verification_code: str | None,
    sessionid: str | None,
) -> dict[str, Any]:
    has_auth = bool(sessionid or (username and password) or settings_path)
    if not has_auth:
        return {
            "available": False,
            "source": "instagrapi",
            "skipped": True,
            "error": "No explicit Instagrapi session, sessionid, or username/password was provided.",
        }

    try:
        instagrapi = load_instagrapi()
        client = instagrapi.Client()
        if settings_path and settings_path.exists():
            client.set_settings(client.load_settings(settings_path))

        if sessionid:
            client.login_by_sessionid(sessionid)
        elif username and password:
            client.login(username, password, verification_code=verification_code or "")
            if settings_path:
                settings_path.parent.mkdir(parents=True, exist_ok=True)
                client.dump_settings(settings_path)

        shortcode = extract_instagram_shortcode(url)
        if not media_id and not shortcode:
            raise ExtractorError("Could not parse Instagram shortcode for Instagrapi media lookup.")
        resolved_media_id = media_id or str(client.media_pk_from_code(shortcode))
        media_info = safe_model_dump(client.media_info_v1(resolved_media_id))
        comments: list[dict[str, Any]] = []
        comment_fetch: dict[str, Any] = {"requested": fetch_comments, "strategy": None}
        if fetch_comments:
            comment_start = time_module.monotonic()
            if fetch_comment_replies:
                top_level_comments = client.media_comments(resolved_media_id, amount=max_comments)
                comment_fetch = {
                    "requested": True,
                    "strategy": "instagrapi_top_level_plus_inline_replies",
                    "top_level_count": len(top_level_comments),
                    "reply_count": 0,
                }
                for comment in top_level_comments:
                    normalized_comment = normalize_instagrapi_comment(comment)
                    comments.append(normalized_comment)
                    if comment_time_budget_seconds and time_module.monotonic() - comment_start >= comment_time_budget_seconds:
                        comment_fetch["time_budget_exhausted"] = True
                        break
                    replies = fetch_instagrapi_comment_replies(
                        client,
                        resolved_media_id,
                        str(normalized_comment.get("id") or normalized_comment.get("pk")),
                        max_comment_replies,
                    )
                    comment_fetch["reply_count"] += len(replies)
                    comments.extend(replies)
            else:
                comments, comment_fetch = fetch_fast_instagram_comments(
                    client,
                    resolved_media_id,
                    max_comments=max_comments,
                    time_budget_seconds=comment_time_budget_seconds,
                )
            comment_fetch["elapsed_seconds"] = round(time_module.monotonic() - comment_start, 3)
        user_info = None
        media_user = media_info.get("user") if isinstance(media_info, dict) else None
        user_pk = media_user.get("pk") if isinstance(media_user, dict) else None
        if user_pk:
            try:
                user_info = safe_model_dump(client.user_info_v1(str(user_pk)))
            except Exception as exc:
                user_info = {"error": str(exc)}

        return make_json_safe(
            {
                "available": True,
                "source": "instagrapi",
                "media_id": resolved_media_id,
                "media_info": media_info,
                "user_info": user_info,
                "comment_objects": comments,
                "comment_object_count": len(comments),
                "comment_fetch": comment_fetch,
            }
        )
    except Exception as exc:
        return {"available": False, "source": "instagrapi", "error": str(exc)}


def fetch_instagrapi_comment_replies(
    client: Any,
    media_id: str,
    parent_comment_id: str,
    max_comment_replies: int,
) -> list[dict[str, Any]]:
    if not parent_comment_id:
        return []

    try:
        replies = fetch_raw_instagram_child_comments(client, media_id, parent_comment_id, max_comment_replies)
    except Exception as exc:
        return [
            {
                "source": "instagrapi_reply_error",
                "parent_comment_id": parent_comment_id,
                "error": str(exc),
            }
        ]

    return [
        {
            **reply,
            "source": "instagrapi_reply",
            "parent_comment_id": parent_comment_id,
            "is_reply": True,
        }
        for reply in replies
    ]


def fetch_fast_instagram_comments(
    client: Any,
    media_id: str,
    *,
    max_comments: int,
    time_budget_seconds: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    endpoint = f"media/{media_id}/comments/"
    comments: list[dict[str, Any]] = []
    seen: set[str] = set()
    top_level_count = 0
    preview_reply_count = 0
    page_count = 0
    next_min_id: str | None = None
    started_at = time_module.monotonic()

    while True:
        if time_budget_seconds and time_module.monotonic() - started_at >= time_budget_seconds:
            return comments, {
                "requested": True,
                "strategy": "raw_private_top_level_with_preview_replies",
                "top_level_count": top_level_count,
                "preview_reply_count": preview_reply_count,
                "page_count": page_count,
                "time_budget_exhausted": True,
            }

        params = {"min_id": next_min_id} if next_min_id else None
        result = client.private_request(endpoint, params=params)
        page_count += 1
        raw_comments = result.get("comments", []) or []

        for raw_comment in raw_comments:
            if max_comments and top_level_count >= max_comments:
                break
            normalized = normalize_instagram_raw_comment(raw_comment)
            normalized["source"] = "instagrapi"
            comment_id = str(normalized.get("id") or normalized.get("pk") or "")
            if comment_id and comment_id not in seen:
                seen.add(comment_id)
                comments.append(normalized)
                top_level_count += 1

            for preview_reply in raw_comment.get("preview_child_comments", []) or []:
                reply = normalize_instagram_raw_comment(preview_reply)
                reply["source"] = "instagrapi_preview_reply"
                reply["is_reply"] = True
                reply["parent_comment_id"] = comment_id or reply.get("parent_comment_id")
                reply_id = str(reply.get("id") or reply.get("pk") or "")
                if reply_id and reply_id not in seen:
                    seen.add(reply_id)
                    comments.append(reply)
                    preview_reply_count += 1

        if max_comments and top_level_count >= max_comments:
            break

        if not (result.get("has_more_headload_comments") and result.get("next_min_id")):
            break
        next_min_id = result.get("next_min_id")

    return comments, {
        "requested": True,
        "strategy": "raw_private_top_level_with_preview_replies",
        "top_level_count": top_level_count,
        "preview_reply_count": preview_reply_count,
        "page_count": page_count,
        "time_budget_exhausted": False,
    }


def fetch_raw_instagram_child_comments(
    client: Any,
    media_id: str,
    parent_comment_id: str,
    max_comment_replies: int,
) -> list[dict[str, Any]]:
    endpoint = f"media/{media_id}/comments/{parent_comment_id}/inline_child_comments/"
    replies: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_comments(raw_comments: list[dict[str, Any]]) -> None:
        for raw_comment in raw_comments:
            normalized = normalize_instagram_raw_comment(raw_comment)
            comment_id = str(normalized.get("id") or normalized.get("pk") or "")
            if comment_id and comment_id in seen:
                continue
            if comment_id:
                seen.add(comment_id)
            replies.append(normalized)

    result = client.private_request(endpoint)
    add_comments(result.get("child_comments", []) or [])

    cursor = result.get("next_min_child_cursor")
    while result.get("has_more_head_child_comments") and cursor:
        if max_comment_replies and len(replies) >= max_comment_replies:
            break
        result = client.private_request(endpoint, params={"min_id": cursor})
        add_comments(result.get("child_comments", []) or [])
        cursor = result.get("next_min_child_cursor")

    cursor = result.get("next_max_child_cursor")
    while result.get("has_more_tail_child_comments") and cursor:
        if max_comment_replies and len(replies) >= max_comment_replies:
            break
        result = client.private_request(endpoint, params={"max_id": cursor})
        add_comments(result.get("child_comments", []) or [])
        cursor = result.get("next_max_child_cursor")

    if max_comment_replies:
        replies = replies[:max_comment_replies]
    return replies


def normalize_instagram_raw_comment(raw_comment: dict[str, Any]) -> dict[str, Any]:
    user = raw_comment.get("user")
    created_at_utc = raw_comment.get("created_at_utc") or raw_comment.get("created_at")
    if isinstance(created_at_utc, (int, float)):
        created_at_utc = dt.datetime.fromtimestamp(created_at_utc, tz=dt.timezone.utc).isoformat()
    like_count = first_not_none(raw_comment.get("comment_like_count"), raw_comment.get("like_count"))

    return make_json_safe(
        {
            "id": raw_comment.get("pk") or raw_comment.get("id"),
            "pk": raw_comment.get("pk") or raw_comment.get("id"),
            "text": raw_comment.get("text"),
            "created_at_utc": created_at_utc,
            "content_type": raw_comment.get("content_type"),
            "status": raw_comment.get("status"),
            "replied_to_comment_id": raw_comment.get("replied_to_comment_id"),
            "has_liked": raw_comment.get("has_liked_comment"),
            "like_count": like_count,
            "likes_count": like_count,
            "owner": user,
            "raw": raw_comment,
        }
    )


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def safe_model_dump(value: Any) -> Any:
    try:
        return value.model_dump(mode="json")
    except Exception:
        pass
    try:
        return value.dict()
    except Exception:
        pass
    return value


def normalize_instagrapi_comment(comment: Any) -> dict[str, Any]:
    payload = safe_model_dump(comment)
    if not isinstance(payload, dict):
        payload = {"raw": payload}

    user = payload.get("user")
    if not isinstance(user, dict):
        user = safe_model_dump(user) if user is not None else None

    return make_json_safe(
        {
            "source": "instagrapi",
            "id": payload.get("pk") or payload.get("id"),
            "pk": payload.get("pk"),
            "text": payload.get("text"),
            "created_at_utc": payload.get("created_at_utc"),
            "content_type": payload.get("content_type"),
            "status": payload.get("status"),
            "replied_to_comment_id": payload.get("replied_to_comment_id"),
            "has_liked": payload.get("has_liked"),
            "like_count": payload.get("like_count"),
            "likes_count": payload.get("like_count"),
            "owner": user,
            "raw": payload,
        }
    )


def safe_getattr(obj: Any, name: str) -> Any:
    try:
        value = getattr(obj, name)
        if callable(value):
            return value()
        return value
    except Exception:
        return None


def isoformat_or_none(value: Any) -> str | None:
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return None


def normalize_instaloader_profile(profile: Any) -> dict[str, Any] | None:
    if profile is None:
        return None

    fields = [
        "username",
        "userid",
        "full_name",
        "biography",
        "external_url",
        "followers",
        "followees",
        "mediacount",
        "is_private",
        "is_verified",
        "profile_pic_url",
    ]
    return {field: safe_getattr(profile, field) for field in fields}


def normalize_instaloader_location(location: Any) -> dict[str, Any] | None:
    if location is None:
        return None
    fields = ["id", "name", "slug", "lat", "lng"]
    return {field: safe_getattr(location, field) for field in fields}


def normalize_instaloader_sidecar(post: Any) -> list[dict[str, Any]]:
    try:
        nodes = list(post.get_sidecar_nodes())
    except Exception:
        return []

    normalized = []
    for node in nodes:
        normalized.append(
            {
                "is_video": safe_getattr(node, "is_video"),
                "display_url": safe_getattr(node, "display_url"),
                "video_url": safe_getattr(node, "video_url"),
            }
        )
    return normalized


def normalize_instaloader_comments(post: Any, max_comments: int) -> list[dict[str, Any]]:
    try:
        comments_iter = post.get_comments()
    except Exception:
        return []

    comments = []
    for index, comment in enumerate(comments_iter):
        if max_comments > 0 and index >= max_comments:
            break
        owner = safe_getattr(comment, "owner")
        comments.append(
            {
                "id": safe_getattr(comment, "id"),
                "created_at_utc": isoformat_or_none(safe_getattr(comment, "created_at_utc")),
                "text": safe_getattr(comment, "text"),
                "likes_count": safe_getattr(comment, "likes_count"),
                "owner": normalize_instaloader_profile(owner),
                "answers": normalize_instaloader_comment_answers(comment),
            }
        )
    return comments


def normalize_instaloader_comment_answers(comment: Any) -> list[dict[str, Any]]:
    try:
        answers_iter = comment.answers
    except Exception:
        return []

    answers = []
    for answer in answers_iter:
        owner = safe_getattr(answer, "owner")
        answers.append(
            {
                "id": safe_getattr(answer, "id"),
                "created_at_utc": isoformat_or_none(safe_getattr(answer, "created_at_utc")),
                "text": safe_getattr(answer, "text"),
                "likes_count": safe_getattr(answer, "likes_count"),
                "owner": normalize_instaloader_profile(owner),
            }
        )
    return answers


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def merge_hashtags(info: dict[str, Any], supplement: dict[str, Any] | None) -> list[str]:
    hashtags = set(collect_hashtags(info))
    if supplement and supplement.get("available"):
        for key in ["caption_hashtags", "hashtags"]:
            for tag in supplement.get(key) or []:
                if isinstance(tag, str) and tag.strip():
                    hashtags.add(tag.strip().lstrip("#"))
        caption = supplement.get("caption")
        if isinstance(caption, str):
            for match in re.findall(r"(?<!\w)#([\w.]+)", caption, flags=re.UNICODE):
                hashtags.add(match.strip("."))
    return sorted(hashtags, key=str.lower)


def supplement_value(supplement: dict[str, Any] | None, path: list[str]) -> Any:
    current: Any = supplement
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if current not in (None, "", [], {}) else None


def normalize_thumbnails(info: dict[str, Any], supplement: dict[str, Any] | None) -> list[dict[str, Any]]:
    thumbnails: list[dict[str, Any]] = []
    for thumb in info.get("thumbnails") or []:
        if not isinstance(thumb, dict) or not thumb.get("url"):
            continue
        thumbnails.append(
            {
                "url": thumb.get("url"),
                "id": thumb.get("id"),
                "width": thumb.get("width"),
                "height": thumb.get("height"),
                "resolution": thumb.get("resolution"),
                "preference": thumb.get("preference"),
            }
        )

    for key in ["thumbnail", "display_url"]:
        url = info.get(key) or supplement_value(supplement, [key])
        if isinstance(url, str) and url and not any(thumb["url"] == url for thumb in thumbnails):
            thumbnails.append({"url": url, "id": key, "width": None, "height": None, "resolution": None, "preference": None})

    for key in ["display_url", "video_url"]:
        url = supplement_value(supplement, [key])
        if isinstance(url, str) and url and not any(thumb["url"] == url for thumb in thumbnails):
            thumbnails.append({"url": url, "id": f"instaloader_{key}", "width": None, "height": None, "resolution": None, "preference": None})

    instagrapi_thumbnail = supplement_value(supplement, ["instagrapi", "media_info", "thumbnail_url"])
    if isinstance(instagrapi_thumbnail, str) and instagrapi_thumbnail and not any(thumb["url"] == instagrapi_thumbnail for thumb in thumbnails):
        thumbnails.append(
            {
                "url": instagrapi_thumbnail,
                "id": "instagrapi_thumbnail_url",
                "width": None,
                "height": None,
                "resolution": None,
                "preference": None,
            }
        )

    image_candidates = supplement_value(supplement, ["instagrapi", "media_info", "image_versions2", "candidates"])
    if isinstance(image_candidates, list):
        for index, candidate in enumerate(image_candidates):
            if not isinstance(candidate, dict) or not candidate.get("url"):
                continue
            if any(thumb["url"] == candidate["url"] for thumb in thumbnails):
                continue
            thumbnails.append(
                {
                    "url": candidate.get("url"),
                    "id": f"instagrapi_image_candidate_{index}",
                    "width": candidate.get("width"),
                    "height": candidate.get("height"),
                    "resolution": f"{candidate.get('width')}x{candidate.get('height')}"
                    if candidate.get("width") and candidate.get("height")
                    else None,
                    "preference": None,
                }
            )

    return thumbnails


def normalize_formats(info: dict[str, Any]) -> list[dict[str, Any]]:
    formats = []
    for fmt in info.get("formats") or []:
        if not isinstance(fmt, dict):
            continue
        formats.append(
            {
                "format_id": fmt.get("format_id"),
                "format_note": fmt.get("format_note"),
                "url": fmt.get("url"),
                "ext": fmt.get("ext"),
                "protocol": fmt.get("protocol"),
                "width": fmt.get("width"),
                "height": fmt.get("height"),
                "resolution": fmt.get("resolution"),
                "fps": fmt.get("fps"),
                "vcodec": fmt.get("vcodec"),
                "acodec": fmt.get("acodec"),
                "audio_ext": fmt.get("audio_ext"),
                "video_ext": fmt.get("video_ext"),
                "filesize": fmt.get("filesize"),
                "filesize_approx": fmt.get("filesize_approx"),
                "tbr": fmt.get("tbr"),
                "abr": fmt.get("abr"),
                "vbr": fmt.get("vbr"),
            }
        )
    return formats


def normalize_comments(info: dict[str, Any], supplement: dict[str, Any] | None) -> list[dict[str, Any]]:
    comments = []
    comment_indexes: dict[str, int] = {}

    def add_comment(comment: dict[str, Any], source: str) -> None:
        normalized = make_json_safe(comment)
        if not isinstance(normalized, dict):
            normalized = {"raw": normalized}
        normalized.setdefault("source", source)
        key = str(normalized.get("id") or normalized.get("pk") or normalized.get("comment_id") or "")
        if not key:
            owner = normalized.get("author") or normalized.get("owner") or normalized.get("user") or ""
            key = f"{source}:{owner}:{normalized.get('text')}"
        if key in comment_indexes:
            existing = comments[comment_indexes[key]]
            existing_sources = existing.setdefault("sources", [existing.get("source")])
            if source not in existing_sources:
                existing_sources.append(source)
            for field, value in normalized.items():
                if value in (None, "", [], {}):
                    continue
                if existing.get(field) in (None, "", [], {}):
                    existing[field] = value
                elif field in {"like_count", "likes_count", "owner", "raw"} and source.startswith("instagrapi"):
                    existing[field] = value
            has_richer_instagram_fields = any(
                normalized.get(field) not in (None, "", [], {}) for field in ["like_count", "likes_count", "owner", "raw"]
            )
            if source.startswith("instagrapi") and has_richer_instagram_fields:
                existing["source"] = source
            return
        comment_indexes[key] = len(comments)
        normalized.setdefault("sources", [source])
        comments.append(normalized)

    for comment in info.get("comments") or []:
        if isinstance(comment, dict):
            add_comment(comment, "yt-dlp")
    if supplement and supplement.get("available"):
        for comment in supplement.get("comment_objects") or []:
            add_comment(comment, "instaloader")
    instagrapi_supplement = supplement_value(supplement, ["instagrapi"])
    if isinstance(instagrapi_supplement, dict) and instagrapi_supplement.get("available"):
        for comment in instagrapi_supplement.get("comment_objects") or []:
            add_comment(comment, "instagrapi")
    return comments


def comment_source_summary(info: dict[str, Any], supplement: dict[str, Any] | None) -> dict[str, Any]:
    instagrapi_supplement = supplement_value(supplement, ["instagrapi"])
    return {
        "yt_dlp": {
            "count": len(info.get("comments") or []),
            "available": bool(info.get("comments")),
        },
        "instaloader": {
            "count": len((supplement or {}).get("comment_objects") or []),
            "available": bool(supplement and supplement.get("available")),
            "error": (supplement or {}).get("error"),
        },
        "instagrapi": {
            "count": len((instagrapi_supplement or {}).get("comment_objects") or [])
            if isinstance(instagrapi_supplement, dict)
            else 0,
            "available": bool(isinstance(instagrapi_supplement, dict) and instagrapi_supplement.get("available")),
            "skipped": bool(isinstance(instagrapi_supplement, dict) and instagrapi_supplement.get("skipped")),
            "error": instagrapi_supplement.get("error") if isinstance(instagrapi_supplement, dict) else None,
        },
    }


def normalize_video_info(
    *,
    platform: str,
    info: dict[str, Any],
    transcript: TranscriptResult,
    supplemental_metadata: dict[str, Any] | None,
    include_raw: bool,
) -> dict[str, Any]:
    thumbnails = normalize_thumbnails(info, supplemental_metadata)
    media_formats = normalize_formats(info)
    public_comments = normalize_comments(info, supplemental_metadata)
    comment_sources = comment_source_summary(info, supplemental_metadata)
    follower_count = first_present(
        info,
        [
            "channel_follower_count",
            "uploader_follower_count",
            "follower_count",
            "channel_subscriber_count",
        ],
    ) or supplement_value(supplemental_metadata, ["owner_profile", "followers"]) or supplement_value(
        supplemental_metadata, ["instagrapi", "user_info", "follower_count"]
    )
    views = (
        info.get("view_count")
        or supplement_value(supplemental_metadata, ["video_view_count"])
        or supplement_value(supplemental_metadata, ["instagrapi", "media_info", "view_count"])
        or supplement_value(supplemental_metadata, ["instagrapi", "media_info", "play_count"])
    )
    likes = (
        info.get("like_count")
        or supplement_value(supplemental_metadata, ["likes"])
        or supplement_value(supplemental_metadata, ["instagrapi", "media_info", "like_count"])
    )
    comments = (
        info.get("comment_count")
        or supplement_value(supplemental_metadata, ["comments"])
        or supplement_value(supplemental_metadata, ["instagrapi", "media_info", "comment_count"])
    )
    creator = first_present(info, ["uploader", "channel", "creator", "uploader_id", "channel_id"]) or supplement_value(
        supplemental_metadata, ["instagrapi", "media_info", "user", "full_name"]
    ) or supplement_value(supplemental_metadata, ["instagrapi", "media_info", "user", "username"])
    creator_id = first_present(info, ["uploader_id", "channel_id", "creator_id"]) or supplement_value(
        supplemental_metadata, ["instagrapi", "media_info", "user", "pk"]
    )
    description = info.get("description") or supplement_value(
        supplemental_metadata, ["instagrapi", "media_info", "caption_text"]
    )

    video = {
        "platform": platform,
        "id": info.get("id"),
        "url": info.get("webpage_url") or info.get("original_url"),
        "title": info.get("title"),
        "description": description,
        "creator": creator,
        "creator_id": creator_id,
        "creator_url": first_present(info, ["uploader_url", "channel_url"]),
        "follower_count": follower_count,
        "hashtags": merge_hashtags(info, supplemental_metadata),
        "upload_date": iso_upload_date(info),
        "duration_seconds": info.get("duration"),
        "views": views,
        "likes": likes,
        "comments": comments,
        "thumbnail": info.get("thumbnail") or supplement_value(supplemental_metadata, ["display_url"]),
        "thumbnail_urls": [thumb["url"] for thumb in thumbnails if thumb.get("url")],
        "thumbnails": thumbnails,
        "media_formats": media_formats,
        "media_format_count": len(media_formats),
        "public_comment_objects": public_comments,
        "public_comment_object_count": len(public_comments),
        "comment_object_gap": comments - len(public_comments) if isinstance(comments, int) else None,
        "comment_sources": comment_sources,
        "extra": {
            "extractor": info.get("extractor"),
            "extractor_key": info.get("extractor_key"),
            "display_id": info.get("display_id"),
            "webpage_url_domain": info.get("webpage_url_domain"),
            "availability": info.get("availability"),
            "age_limit": info.get("age_limit"),
            "categories": info.get("categories"),
            "tags": info.get("tags"),
            "width": info.get("width"),
            "height": info.get("height"),
            "resolution": info.get("resolution"),
            "fps": info.get("fps"),
            "aspect_ratio": info.get("aspect_ratio"),
            "vcodec": info.get("vcodec"),
            "acodec": info.get("acodec"),
            "video_ext": info.get("video_ext"),
            "audio_ext": info.get("audio_ext"),
            "format_id": info.get("format_id"),
            "format_note": info.get("format_note"),
            "filesize": info.get("filesize"),
            "filesize_approx": info.get("filesize_approx"),
            "chapters": info.get("chapters"),
        },
        "supplemental_metadata": supplemental_metadata,
        "transcript": asdict(transcript),
    }

    if include_raw:
        video["raw_metadata"] = info

    return video


def instagram_auth_available(
    *,
    platform: str,
    use_instagrapi: bool,
    instagrapi_settings: Path | None,
    instagram_username: str | None,
    instagram_password: str | None,
    instagram_sessionid: str | None,
) -> bool:
    return bool(
        is_instagram_media(platform)
        and use_instagrapi
        and (instagram_sessionid or (instagram_username and instagram_password) or instagrapi_settings)
    )


def build_instagram_supplemental_metadata(
    *,
    url: str,
    media_id: str | None,
    fetch_comments: bool,
    max_comments: int,
    use_instaloader: bool,
    instaloader_session_user: str | None,
    use_instagrapi: bool,
    instagrapi_settings: Path | None,
    instagram_username: str | None,
    instagram_password: str | None,
    instagram_verification_code: str | None,
    instagram_sessionid: str | None,
    fetch_comment_replies: bool,
    max_comment_replies: int,
    comment_time_budget_seconds: float | None,
) -> dict[str, Any] | None:
    platform = detect_platform(url) or "instagram"
    auth_available = instagram_auth_available(
        platform=platform,
        use_instagrapi=use_instagrapi,
        instagrapi_settings=instagrapi_settings,
        instagram_username=instagram_username,
        instagram_password=instagram_password,
        instagram_sessionid=instagram_sessionid,
    )
    supplemental_metadata: dict[str, Any] | None = None

    if use_instaloader and not auth_available:
        supplemental_metadata = fetch_instagram_supplement(
            url,
            fetch_comments=fetch_comments,
            max_comments=max_comments,
            session_user=instaloader_session_user,
        )
    elif use_instaloader and auth_available:
        supplemental_metadata = {
            "available": False,
            "source": "instaloader",
            "skipped": True,
            "reason": "Skipped because authenticated Instagrapi is available for faster Instagram comments and metadata.",
        }

    if use_instagrapi:
        if supplemental_metadata is None:
            supplemental_metadata = {"available": False, "source": "instaloader", "skipped": True}
        supplemental_metadata["instagrapi"] = fetch_instagram_instagrapi_supplement(
            url,
            media_id=media_id,
            fetch_comments=fetch_comments,
            fetch_comment_replies=fetch_comment_replies,
            max_comments=max_comments,
            max_comment_replies=max_comment_replies,
            comment_time_budget_seconds=comment_time_budget_seconds,
            settings_path=instagrapi_settings,
            username=instagram_username,
            password=instagram_password,
            verification_code=instagram_verification_code,
            sessionid=instagram_sessionid,
        )

    return supplemental_metadata


def extract_pair(
    youtube_url: str,
    instagram_url: str,
    *,
    language: str,
    cookies: str | None,
    cookies_from_browser: str | None,
    fetch_comments: bool = False,
    max_comments: int = 500,
    use_instaloader: bool = True,
    instaloader_session_user: str | None = None,
    use_instagrapi: bool = True,
    instagrapi_settings: Path | None = None,
    instagram_username: str | None = None,
    instagram_password: str | None = None,
    instagram_verification_code: str | None = None,
    instagram_sessionid: str | None = None,
    fetch_comment_replies: bool = False,
    max_comment_replies: int = 0,
    comment_time_budget_seconds: float | None = None,
    transcribe_missing: bool = True,
    asr_provider: str = "auto",
    asr_model: str = "base",
    hf_asr_model: str = "openai/whisper-large-v3-turbo",
    hf_token: str | None = None,
    asr_timeout_seconds: float = 60,
    asr_language: str | None = None,
    asr_device: str = "cpu",
    asr_compute_type: str = "int8",
    asr_cache_dir: Path | None = None,
    keep_audio: bool = False,
    include_raw: bool = False,
) -> dict[str, Any]:
    validate_urls(youtube_url, instagram_url)

    cache_dir = asr_cache_dir or Path(".cache") / "social_video_extractor"
    video_jobs = [("youtube", youtube_url), (detect_platform(instagram_url) or "instagram", instagram_url)]

    def process_video(platform: str, url: str) -> dict[str, Any]:
        started_at = time_module.monotonic()
        timing: dict[str, float] = {}
        auth_available = instagram_auth_available(
            platform=platform,
            use_instagrapi=use_instagrapi,
            instagrapi_settings=instagrapi_settings,
            instagram_username=instagram_username,
            instagram_password=instagram_password,
            instagram_sessionid=instagram_sessionid,
        )

        supplemental_future: concurrent.futures.Future[dict[str, Any] | None] | None = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as inner_executor:
            if is_instagram_media(platform) and auth_available:
                supplemental_future = inner_executor.submit(
                    build_instagram_supplemental_metadata,
                    url=url,
                    media_id=None,
                    fetch_comments=fetch_comments,
                    max_comments=max_comments,
                    use_instaloader=use_instaloader,
                    instaloader_session_user=instaloader_session_user,
                    use_instagrapi=use_instagrapi,
                    instagrapi_settings=instagrapi_settings,
                    instagram_username=instagram_username,
                    instagram_password=instagram_password,
                    instagram_verification_code=instagram_verification_code,
                    instagram_sessionid=instagram_sessionid,
                    fetch_comment_replies=fetch_comment_replies,
                    max_comment_replies=max_comment_replies,
                    comment_time_budget_seconds=comment_time_budget_seconds,
                )

            stage_started = time_module.monotonic()
            info = extract_info(
                url,
                cookies=cookies,
                cookies_from_browser=cookies_from_browser,
                fetch_comments=fetch_comments and not (is_instagram_media(platform) and auth_available),
            )
            timing["base_metadata_seconds"] = round(time_module.monotonic() - stage_started, 3)

            if is_instagram_media(platform) and supplemental_future is None:
                media_id = None
                supplemental_metadata = build_instagram_supplemental_metadata(
                    url=url,
                    media_id=media_id,
                    fetch_comments=fetch_comments,
                    max_comments=max_comments,
                    use_instaloader=use_instaloader,
                    instaloader_session_user=instaloader_session_user,
                    use_instagrapi=use_instagrapi,
                    instagrapi_settings=instagrapi_settings,
                    instagram_username=instagram_username,
                    instagram_password=instagram_password,
                    instagram_verification_code=instagram_verification_code,
                    instagram_sessionid=instagram_sessionid,
                    fetch_comment_replies=fetch_comment_replies,
                    max_comment_replies=max_comment_replies,
                    comment_time_budget_seconds=comment_time_budget_seconds,
                )
                timing["supplemental_metadata_seconds"] = 0.0
            else:
                supplemental_metadata = None

            stage_started = time_module.monotonic()
            transcript_future = inner_executor.submit(
                resolve_transcript,
                info=info,
                url=url,
                platform=platform,
                preferred_caption_language=language,
                transcribe_missing=transcribe_missing,
                asr_provider=asr_provider,
                asr_model=asr_model,
                hf_asr_model=hf_asr_model,
                hf_token=hf_token,
                asr_timeout_seconds=asr_timeout_seconds,
                asr_language=asr_language,
                asr_device=asr_device,
                asr_compute_type=asr_compute_type,
                asr_cache_dir=cache_dir,
                keep_audio=keep_audio,
                cookies=cookies,
                cookies_from_browser=cookies_from_browser,
            )

            if supplemental_future is not None:
                supplement_started = time_module.monotonic()
                supplemental_metadata = supplemental_future.result()
                timing["supplemental_metadata_wait_seconds"] = round(time_module.monotonic() - supplement_started, 3)

            transcript = transcript_future.result()
            timing["transcript_wait_seconds"] = round(time_module.monotonic() - stage_started, 3)

        video = normalize_video_info(
            platform=platform,
            info=info,
            transcript=transcript,
            supplemental_metadata=supplemental_metadata,
            include_raw=include_raw,
        )
        timing["total_video_seconds"] = round(time_module.monotonic() - started_at, 3)
        video["timing"] = timing
        return video

    videos_by_platform: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_video, platform, url): platform for platform, url in video_jobs}
        for future in concurrent.futures.as_completed(futures):
            platform = futures[future]
            videos_by_platform[platform] = future.result()

    videos = [videos_by_platform[platform] for platform, _ in video_jobs]

    return {
        "requested_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "videos": videos,
    }


def validate_urls(youtube_url: str, instagram_url: str) -> None:
    if detect_platform(youtube_url) != "youtube":
        raise URLValidationError("A valid YouTube URL is mandatory.")
    if not extract_youtube_video_id(youtube_url):
        raise URLValidationError("The YouTube URL must point to a specific video.")
    if not is_instagram_media(detect_platform(instagram_url)):
        raise URLValidationError("A valid Instagram Reel or post URL is mandatory.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pull transcripts and metadata for one YouTube video and one Instagram media URL."
    )
    parser.add_argument("urls", nargs="*", help="Optional positional URLs: one YouTube and one Instagram media URL.")
    parser.add_argument("--youtube-url", help="YouTube video URL. Mandatory when using named inputs.")
    parser.add_argument("--instagram-url", help="Instagram Reel or post URL. Mandatory when using named inputs.")
    parser.add_argument("--language", default="en", help="Preferred transcript language, default: en.")
    parser.add_argument("--cookies", help="Path to a Netscape cookies.txt file for authenticated extraction.")
    parser.add_argument(
        "--cookies-from-browser",
        help="Browser name for yt-dlp cookie import, for example: chrome, edge, firefox.",
    )
    parser.add_argument("--output", "-o", help="Write JSON output to this path. Prints to stdout if omitted.")
    parser.add_argument("--include-raw", action="store_true", help="Include the raw yt-dlp metadata in output.")
    parser.add_argument(
        "--fetch-comments",
        action="store_true",
        help="Fetch public comment objects when the platform extractor supports it.",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=500,
        help="Maximum Instagram comments to fetch with Instaloader; use 0 for no limit. Default: 500.",
    )
    parser.add_argument(
        "--no-instaloader",
        action="store_false",
        dest="use_instaloader",
        help="Skip the Instagram-specific Instaloader supplement.",
    )
    parser.add_argument(
        "--instaloader-session-user",
        help="Load an Instaloader browser/session file for this Instagram username.",
    )
    parser.add_argument(
        "--no-instagrapi",
        action="store_false",
        dest="use_instagrapi",
        help="Skip the authenticated Instagrapi comment/media supplement.",
    )
    parser.add_argument(
        "--instagrapi-settings",
        help="Path to an Instagrapi settings JSON file to load or update after login.",
    )
    parser.add_argument("--instagram-username", help="Instagram username for Instagrapi login.")
    parser.add_argument("--instagram-password", help="Instagram password for Instagrapi login.")
    parser.add_argument("--instagram-verification-code", help="2FA verification code for Instagrapi login.")
    parser.add_argument("--instagram-sessionid", help="Explicit Instagram sessionid for Instagrapi login.")
    parser.add_argument(
        "--fetch-comment-replies",
        action="store_true",
        help="Fetch Instagram threaded replies for each comment through Instagrapi.",
    )
    parser.add_argument(
        "--max-comment-replies",
        type=int,
        default=0,
        help="Maximum replies per Instagram comment; use 0 for no limit. Default: 0.",
    )
    parser.add_argument(
        "--comment-time-budget-seconds",
        type=float,
        help="Best-effort time budget for Instagram comment fetching; useful with --max-comments 0.",
    )
    parser.add_argument(
        "--no-asr",
        action="store_false",
        dest="transcribe_missing",
        help="Do not transcribe downloaded audio when platform captions are unavailable.",
    )
    parser.add_argument(
        "--asr-provider",
        default="auto",
        choices=["auto", "local", "hf"],
        help="ASR provider for missing captions. auto uses Hugging Face when a token is available, else local.",
    )
    parser.add_argument("--asr-model", default="base", help="faster-whisper model name. Default: base.")
    parser.add_argument(
        "--hf-asr-model",
        default="openai/whisper-large-v3-turbo",
        help="Hosted Hugging Face ASR model. Default: openai/whisper-large-v3-turbo.",
    )
    parser.add_argument("--hf-token", help="Hugging Face token. Prefer HF_TOKEN env var to avoid shell history.")
    parser.add_argument("--asr-timeout-seconds", type=float, default=60, help="Hosted ASR timeout. Default: 60.")
    parser.add_argument("--asr-language", help="Force ASR language; omit for automatic language detection.")
    parser.add_argument("--asr-device", default="cpu", help="faster-whisper device. Default: cpu.")
    parser.add_argument("--asr-compute-type", default="int8", help="faster-whisper compute type. Default: int8.")
    parser.add_argument(
        "--asr-cache-dir",
        default=str(Path(".cache") / "social_video_extractor"),
        help="Directory for temporary ASR media downloads.",
    )
    parser.add_argument("--keep-audio", action="store_true", help="Keep downloaded audio/video files used for ASR.")
    parser.add_argument(
        "--require-transcripts",
        action="store_true",
        help="Exit with an error if either video has no readable platform transcript or ASR transcript.",
    )
    parser.set_defaults(use_instaloader=True, use_instagrapi=True, transcribe_missing=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        youtube_url, instagram_url = resolve_input_urls(args)
        result = extract_pair(
            youtube_url,
            instagram_url,
            language=args.language,
            cookies=args.cookies,
            cookies_from_browser=args.cookies_from_browser,
            fetch_comments=args.fetch_comments,
            max_comments=args.max_comments,
            use_instaloader=args.use_instaloader,
            instaloader_session_user=args.instaloader_session_user,
            use_instagrapi=args.use_instagrapi,
            instagrapi_settings=Path(args.instagrapi_settings) if args.instagrapi_settings else None,
            instagram_username=args.instagram_username,
            instagram_password=args.instagram_password,
            instagram_verification_code=args.instagram_verification_code,
            instagram_sessionid=args.instagram_sessionid,
            fetch_comment_replies=args.fetch_comment_replies,
            max_comment_replies=args.max_comment_replies,
            comment_time_budget_seconds=args.comment_time_budget_seconds,
            transcribe_missing=args.transcribe_missing,
            asr_provider=args.asr_provider,
            asr_model=args.asr_model,
            hf_asr_model=args.hf_asr_model,
            hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
            asr_timeout_seconds=args.asr_timeout_seconds,
            asr_language=args.asr_language,
            asr_device=args.asr_device,
            asr_compute_type=args.asr_compute_type,
            asr_cache_dir=Path(args.asr_cache_dir),
            keep_audio=args.keep_audio,
            include_raw=args.include_raw,
        )

        if args.require_transcripts:
            missing = [
                video["platform"]
                for video in result["videos"]
                if not video["transcript"]["available"]
            ]
            if missing:
                raise ExtractorError(f"Missing transcript for: {', '.join(missing)}")

        output = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(output + "\n", encoding="utf-8")
        else:
            print(output)

        return 0
    except (ExtractorError, URLValidationError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
