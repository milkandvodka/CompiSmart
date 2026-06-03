import { newComparisonId, stableIdempotencyKey } from "./format.js";

export const emptyJobForm = {
  comparisonId: newComparisonId(),
  youtubeUrl: "",
  instagramUrl: "",
  embeddingModel: "quality",
  asrModel: "base",
  maxComments: 100,
  requireTranscripts: true,
  commentIntelligence: "evidence",
  creativeFeatures: "evidence",
};

export const embeddingHints = {
  quality: "BGE-M3 is slower but better for multilingual RAG quality.",
  balanced: "E5 base is a middle path: decent multilingual retrieval with less latency.",
  fast: "MiniLM is the fastest option, useful for quick tests, but recall can be weaker.",
};

export const asrModelHints = {
  base: "Fastest local Whisper fallback. Use for quick checks; accuracy can be rough.",
  small: "Better multilingual ASR than base, slower on CPU.",
  medium: "Best local quality option here, but expect a long CPU run.",
};

export function buildExtractIndexPayload(form) {
  return {
    comparison_id: form.comparisonId.trim(),
    youtube_url: form.youtubeUrl.trim(),
    instagram_url: form.instagramUrl.trim(),
    extraction: {
      fetch_comments: true,
      max_comments: Number(form.maxComments) || 100,
      comment_time_budget_seconds: 60,
      instagrapi_settings: ".cache/instagrapi-session.json",
      asr_provider: "auto",
      asr_model: form.asrModel || "base",
      asr_timeout_seconds: 90,
      require_transcripts: form.requireTranscripts,
    },
    index: {
      embedding_model: form.embeddingModel,
      allow_embedding_download: true,
      comment_intelligence: form.commentIntelligence,
      creative_features: form.creativeFeatures,
    },
    idempotency_key: stableIdempotencyKey(form),
  };
}

export function prepareJobForm(form) {
  const next = { ...form };
  const messages = [];
  const errors = [];
  const ytKind = classifySocialUrl(next.youtubeUrl);
  const igKind = classifySocialUrl(next.instagramUrl);

  if (ytKind === "instagram" && igKind === "youtube") {
    [next.youtubeUrl, next.instagramUrl] = [next.instagramUrl, next.youtubeUrl];
    messages.push("The URLs were pasted in the opposite fields, so I swapped them before starting.");
  } else {
    if (ytKind !== "youtube") errors.push("Paste a valid YouTube or YouTube Shorts URL in the YouTube field.");
    if (igKind !== "instagram") errors.push("Paste a valid Instagram Reel/Post URL in the Instagram field.");
  }

  if (!next.comparisonId.trim()) errors.push("Comparison ID is required.");

  return { form: next, messages, errors };
}

export function classifySocialUrl(rawUrl) {
  if (!rawUrl || !rawUrl.trim()) return "missing";
  let url;
  try {
    url = new URL(rawUrl.trim());
  } catch {
    return "invalid";
  }
  const host = url.hostname.toLowerCase().replace(/^www\./, "");
  if (host === "youtu.be" || host.endsWith("youtube.com")) return "youtube";
  if (host.endsWith("instagram.com")) return "instagram";
  return "unknown";
}

export function buildChatPayload({ activeComparison, activeComparisonId, question }) {
  return {
    comparison_id: activeComparisonId,
    question,
    thread_id: `ui_${activeComparisonId}`,
    idempotency_key: stableIdempotencyKey({ activeComparisonId, question }),
    options: {
      llm: "auto",
      embedding_model: activeComparison?.embedding_model || "quality",
      retrieval_mode: "hybrid",
      memory_backend: "local",
      allow_embedding_download: true,
    },
  };
}
