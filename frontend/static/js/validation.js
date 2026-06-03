export const EMBEDDING_HINTS = {
  quality: "BGE-M3 is slower but better for multilingual RAG quality.",
  balanced: "E5 base is a middle path: decent multilingual retrieval with less latency.",
  fast: "MiniLM is the fastest option, useful for quick tests, but recall can be weaker.",
};

export const ASR_MODEL_HINTS = {
  base: "Fastest local Whisper fallback. Use for quick checks; accuracy can be rough.",
  small: "Better multilingual ASR than base, slower on CPU.",
  medium: "Best local quality option here, but expect a long CPU run.",
};

export const DEFAULT_COMMENT_LIMIT = 100;

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
