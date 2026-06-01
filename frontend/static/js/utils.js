export function value(selector) {
  return document.querySelector(selector).value.trim();
}

export function metric(label, value) {
  return `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd>`;
}

export function numberOrUnavailable(value) {
  return value == null ? "unavailable" : Number(value).toLocaleString();
}

export function stableKey(value) {
  return btoa(unescape(encodeURIComponent(JSON.stringify(value)))).slice(0, 96);
}

export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, Number(value) || 0));
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function escapeAttr(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}

export function newComparisonId() {
  const stamp = new Date().toISOString().replaceAll(/[-:.TZ]/g, "").slice(0, 14);
  return `comparison_${stamp}`;
}

export function thumbnailFallback(video) {
  if (video?.platform === "youtube" && video?.source_id) {
    return `https://i.ytimg.com/vi/${encodeURIComponent(video.source_id)}/hqdefault.jpg`;
  }
  return "";
}
