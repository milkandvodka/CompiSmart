export function stableIdempotencyKey(value) {
  return btoa(unescape(encodeURIComponent(JSON.stringify(value)))).slice(0, 96);
}

export function newComparisonId() {
  const stamp = new Date().toISOString().replaceAll(/[-:.TZ]/g, "").slice(0, 14);
  return `comparison_${stamp}`;
}

export function clampPercent(value) {
  if (value == null) return 0;
  return Math.max(0, Math.min(100, Number(value) || 0));
}

export function formatNumber(value) {
  if (value == null) return "unavailable";
  return Number(value).toLocaleString();
}

export function thumbnailFallback(video) {
  if (video?.platform === "youtube" && video?.source_id) {
    return `https://i.ytimg.com/vi/${encodeURIComponent(video.source_id)}/hqdefault.jpg`;
  }
  return "";
}
