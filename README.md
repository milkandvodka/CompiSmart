# Social Video Extractor

Pull normalized transcript and metadata for exactly two required inputs:

- one YouTube video URL
- one Instagram Reel or post URL

## Setup

```powershell
pip install -r requirements.txt
```

## Usage

Named inputs:

```powershell
python social_video_extractor.py `
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" `
  --instagram-url "https://www.instagram.com/reel/REEL_ID/" `
  --include-raw `
  --fetch-comments `
  --output result.json
```

Positional inputs also work in either order:

```powershell
python social_video_extractor.py "https://www.instagram.com/p/POST_ID/" "https://youtu.be/VIDEO_ID"
```

The output contains these normalized fields for both videos:

- transcript text and timestamped segments
- ASR transcript fallback when platform captions are unavailable
- views
- likes
- comments
- creator
- follower count, when exposed by the platform
- hashtags
- upload date
- duration
- primary thumbnail URL and all thumbnail URLs
- media format URLs and technical format metadata

## Authenticated Extraction

Instagram metadata and captions are often unavailable without a logged-in browser session. Use one of these when public extraction fails:

```powershell
python social_video_extractor.py `
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" `
  --instagram-url "https://www.instagram.com/reel/REEL_ID/" `
  --cookies-from-browser chrome
```

or:

```powershell
python social_video_extractor.py `
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" `
  --instagram-url "https://www.instagram.com/reel/REEL_ID/" `
  --cookies "cookies.txt"
```

Use `--require-transcripts` if the run should fail whenever either platform does not expose a readable caption track.

Use `--fetch-comments` with `--include-raw` when you want public comment objects, not only comment counts.

Instagram can expose more fields through the Instaloader supplement, including `video_view_count` and
profile `followers` when public requests are allowed. If anonymous Instagram GraphQL calls are limited,
create/load an Instaloader session and pass:

```powershell
python social_video_extractor.py `
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" `
  --instagram-url "https://www.instagram.com/p/POST_ID/" `
  --instaloader-session-user "YOUR_INSTAGRAM_USERNAME"
```

Full Instagram comment pagination with per-comment like counts usually requires authenticated Instagram
API access. The extractor supports Instagrapi for that:

```powershell
python social_video_extractor.py `
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" `
  --instagram-url "https://www.instagram.com/p/POST_ID/" `
  --fetch-comments `
  --fetch-comment-replies `
  --max-comments 0 `
  --instagram-username "YOUR_INSTAGRAM_USERNAME" `
  --instagram-password "YOUR_INSTAGRAM_PASSWORD" `
  --instagrapi-settings ".cache/instagrapi-session.json"
```

After the first successful login, reuse the saved settings file:

```powershell
python social_video_extractor.py `
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" `
  --instagram-url "https://www.instagram.com/p/POST_ID/" `
  --fetch-comments `
  --fetch-comment-replies `
  --max-comments 0 `
  --instagrapi-settings ".cache/instagrapi-session.json"
```

You can also pass `--instagram-sessionid "SESSION_ID"` if you explicitly provide a session ID.

For higher-quality fallback transcription, use a larger faster-whisper model:

```powershell
python social_video_extractor.py `
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" `
  --instagram-url "https://www.instagram.com/p/POST_ID/" `
  --asr-model small `
  --require-transcripts
```

## Transcript Normalization

Keep the raw ASR transcript, then add Latin-script Hinglish and normalized English variants for RAG:

```powershell
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
python transcript_normalizer.py `
  --input result.json `
  --output result.normalized.json `
  --model gemini-2.5-flash-lite
```

The same Gemini key is used by transcript normalization, index-time analysis, and chat. You can provide it with
`GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_GENAI_API_KEY`, `GEMINI_KEY`, or a local ignored `.env` file.

The normalizer checks for non-English scripts first. English-only transcripts skip the Gemini call and cost nothing.

For local QA only, you can bypass Gemini chat generation and use the installed Codex CLI as a stateless test provider:

```powershell
COMPARAG_LLM_MODE=codex_testing
```

`codex_testing` runs `codex exec` in an empty temporary workspace and passes only the RAG prompt through stdin.
Remove that env setting when you want production behavior to return to Gemini.

Even when exact tools gather metrics or comment lookups, the normal chat path still calls the configured LLM. Tool
results are supplied as evidence/context; they are not used as the final conversational answer. If the LLM provider
fails, the app returns a generation-failure message with the prepared evidence instead of fabricating an answer.

## Local RAG Index

Build the local Chroma index after extraction and transcript normalization:

```powershell
python -m comparag index `
  --input result.normalized.json `
  --comparison-id demo_run `
  --embedding-model balanced `
  --comment-intelligence llm `
  --creative-features llm `
  --analysis-model gemini-2.5-flash-lite `
  --chroma-dir .cache\chroma `
  --app-dir .cache\comparag `
  --allow-embedding-download
```

Use `--allow-embedding-download` only the first time a free local embedding model needs to be cached.
Embedding presets:

- `fast`: `sentence-transformers/all-MiniLM-L6-v2`
- `balanced`: `intfloat/multilingual-e5-base`
- `quality`: `BAAI/bge-m3`

The index command also writes a local BM25 lexical index beside Chroma, so exact wording and semantic meaning are both available.
It also builds cached comment intelligence and transcript-only creative features before chunking. If no Gemini key is
available, those analysis stages store compressed evidence only and record warnings; semantic themes and creative
features require an LLM call.
Repeated indexing is incremental: unchanged chunk fingerprints skip Chroma and embedding work. Use `--force-reindex`
when you intentionally want to rebuild every vector.

Indexing also stores a structured `comment_facts` table beside vector chunks. Exact comment analytics such as
commenter usernames, user IDs, profile URLs, comment-like sums, and phrase counts are exposed to the LLM as tool
evidence, so exact phrase questions do not depend on semantic search.

Ask questions with the LangGraph chat orchestrator:

```powershell
python -m comparag chat `
  --comparison-id demo_run `
  --question "Compare the hooks in the first 5 seconds" `
  --retrieval-mode hybrid `
  --llm gemini
```

Retrieval modes:

- `semantic`: Chroma vector search only
- `lexical`: BM25 exact-word retrieval only
- `hybrid`: semantic + lexical reciprocal-rank fusion

For offline debugging without an LLM call:

```powershell
python -m comparag chat `
  --comparison-id demo_run `
  --question "What's the engagement rate of each?" `
  --llm fallback `
  --no-stream
```

Enable local reranking when you want higher precision over the fused candidates:

```powershell
python -m comparag chat `
  --comparison-id demo_run `
  --question "Suggest improvements for B based on what worked in A" `
  --retrieval-mode hybrid `
  --enable-reranker `
  --reranker-model quality `
  --allow-reranker-download
```

The RAG layer stores exact metrics separately from semantic chunks. Metrics and creator questions answer from
structured data; hook/comparison/improvement questions retrieve balanced chunks from Video A and Video B.
Context budgets are model-aware through provider profiles for Gemini, Codex test mode, OpenAI-style models, and
smaller models. Recent history, exact tool output, retrieved context, per-chunk text, and the final prompt are all
capped before the LLM call.

## FastAPI Backend

Run the backend around the same agent used by the CLI:

```powershell
python -m uvicorn comparag.api.app:app --host 127.0.0.1 --port 8000
```

Useful local endpoints:

- `GET /health`
- `GET /comparisons`
- `GET /comparisons/{comparison_id}`
- `POST /chat`
- `POST /chat/stream` for server-sent token events
- `POST /jobs/index` for background indexing from an extractor JSON path or payload
- `POST /jobs/extract-index` for background URL extraction plus indexing
- `GET /jobs/{job_id}`

OpenAPI docs are available at `http://127.0.0.1:8000/docs`.
The lightweight UI is served by the same backend at `http://127.0.0.1:8000/ui`.

Job responses include a `progress` object and `progress_events` history so the UI can poll `GET /jobs/{job_id}` and
show stages such as `extracting`, `transcribing`, `analysis`, `chunking`, `embedding`, `lexical_index`, and `complete`.

The backend has lightweight in-memory API protection:

- rate limits are applied per client/path, with stricter limits on `/chat` and `/jobs`
- duplicate job submissions are idempotent by payload or `Idempotency-Key`
- duplicate `/chat` requests in a short window return the cached answer instead of appending another memory turn
- duplicate `/chat/stream` requests replay the completed stream when available or return a duplicate event while one is already in progress

## Docker

Build and run the backend container:

```powershell
docker compose build api
docker compose up -d api
```

The compose service exposes the container on `http://127.0.0.1:8001` and serves the static UI at
`http://127.0.0.1:8001/ui/static/index.html`.
It mounts local `.cache` into `/app/.cache`, including Chroma data, Instagram session settings, and Hugging Face model
cache files used by local embeddings.
Secrets stay in the local ignored `.env` file and are loaded at runtime, not baked into the image.
Compose sets `COMPARAG_LLM_MODE=auto` and `COMPARAG_DISABLE_GEMINI=1` for local testing, because Gemini can hang or
quota out during development. With the provided local `.env`, Docker uses OpenAI first, then retrieval fallback.
For production, remove `COMPARAG_DISABLE_GEMINI=1` if you want auto mode to try Gemini before OpenAI.
For local responsiveness, Compose also sets `GEMINI_TIMEOUT_SECONDS=10` and `GEMINI_MAX_RETRIES=0`; raise those values
later if the hosted backend has stable outbound model latency.

## UI

The frontend lives in [frontend](./frontend). The React/Vite app is scaffolded for the final UI:

```powershell
cd frontend
npm install
npm run dev
```

For Docker-backed local testing, create `frontend/.env.local` with:

```powershell
VITE_API_BASE_URL=http://127.0.0.1:8001
```

The currently running local frontend is expected at `http://127.0.0.1:5173`.

The UI opens in a fresh state by default. The visible job form asks only for a comparison ID, one YouTube URL, one
Instagram URL, and the embedding quality preset. If the YouTube and Instagram URLs are pasted into the wrong fields,
the UI swaps them before submission and shows a short message. Comment collection, transcript enforcement, and analysis
modes are handled by backend defaults so the form stays usable.



## Code Structure

Backend orchestration is split by responsibility:

- `comparag/api`: FastAPI routes, job registry, API protection, runtime wiring, indexing service helpers
- `comparag/chat`: LangGraph chat engine, evidence planning, retrieval routing, prompt building, citation validation
- `comparag/chunks`: transcript/comment/analysis chunk builders
- `comparag/comment_tools`: exact comment fact table, phrase parsing, comment lookup tools, citation formatting

## Persistent Memory

Local chat memory is in-process by default. To use Supabase-backed persistent memory, set:

```powershell
SUPABASE_URL="https://YOUR_PROJECT.supabase.co"
SUPABASE_SERVICE_ROLE_KEY="YOUR_SERVICE_ROLE_KEY"
```

Then create the memory tables by running [001_comparag_memory.sql](./supabase/migrations/001_comparag_memory.sql)
in the Supabase SQL editor. The app only performs read/write/upsert operations at runtime; it does not delete tables.

Chat uses `--memory-backend auto` by default. Auto uses Supabase when configured and falls back to local memory if
the tables are not created yet. Use `--memory-backend supabase` when you want missing-table errors to fail loudly.

For long chats, the engine keeps recent turns plus a rolling long-term summary. The summary is updated by the
configured LLM after enough new messages accumulate, stored in `comparag_memory_summaries`, and read back into both
the evidence planner and final answer prompt. If no LLM provider is configured, summaries are not guessed.
