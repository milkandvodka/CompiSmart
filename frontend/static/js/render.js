import { els, state } from "./state.js";
import { clamp, escapeAttr, escapeHtml, metric, numberOrUnavailable, thumbnailFallback } from "./utils.js";

export function renderVideos() {
  const videos = state.activeComparison?.videos || [];
  els.videoCards.innerHTML = "";
  if (!videos.length) {
    els.videoCards.innerHTML = '<div class="empty">No comparison loaded.</div>';
    return;
  }
  for (const video of videos) {
    const card = document.createElement("article");
    card.className = "video-card";
    const fallback = thumbnailFallback(video);
    const imageUrl = video.thumbnail || fallback;
    card.innerHTML = `
      ${imageUrl ? `<img src="${escapeAttr(imageUrl)}" alt="" loading="lazy" data-fallback="${escapeAttr(fallback)}" />` : '<div class="thumbnail-missing">No thumbnail</div>'}
      <div class="video-body">
        <div class="video-title">
          <h3>Video ${escapeHtml(video.video_id)}</h3>
          <span>${escapeHtml(video.platform || "")}</span>
        </div>
        <p class="title">${escapeHtml(video.title || "Untitled")}</p>
        <dl>
          ${metric("Creator", video.creator || "unavailable")}
          ${metric("Followers", numberOrUnavailable(video.follower_count))}
          ${metric("Views", numberOrUnavailable(video.views))}
          ${metric("Likes", numberOrUnavailable(video.likes))}
          ${metric("Comments", numberOrUnavailable(video.comments))}
          ${metric("Fetched comments", numberOrUnavailable(video.fetched_comment_count))}
          ${metric("Engagement", video.engagement_rate == null ? "unavailable" : `${Number(video.engagement_rate).toFixed(2)}%`)}
          ${metric("Duration", video.duration_seconds == null ? "unavailable" : `${Number(video.duration_seconds).toFixed(1)}s`)}
          ${metric("Upload", video.upload_date || "unavailable")}
        </dl>
      </div>
    `;
    els.videoCards.append(card);
    const image = card.querySelector("img");
    if (image) image.addEventListener("error", handleThumbnailError);
  }
}

function handleThumbnailError(event) {
  const image = event.currentTarget;
  const fallback = image.dataset.fallback;
  if (fallback && image.src !== fallback) {
    image.src = fallback;
    image.dataset.fallback = "";
    return;
  }
  image.replaceWith(Object.assign(document.createElement("div"), {
    className: "thumbnail-missing",
    textContent: "Thumbnail unavailable",
  }));
}

export function renderJob() {
  const job = state.job;
  els.runPipeline.disabled = job && ["queued", "running"].includes(job.status);
  els.runPipeline.textContent = els.runPipeline.disabled ? "Working..." : "Run Pipeline";
  if (!job) {
    els.jobBox.innerHTML = "";
    return;
  }
  const percent = clamp(job.progress?.percent || 0, 0, 100);
  els.jobBox.innerHTML = `
    <div class="job">
      <div class="job-row">
        <strong>${escapeHtml(job.status)}</strong>
        <span>${escapeHtml(job.progress?.stage || "queued")}</span>
      </div>
      <div class="bar"><span style="width:${percent}%"></span></div>
      <p>${escapeHtml(job.progress?.message || "Waiting for progress...")}</p>
      <small>${escapeHtml(job.job_id)}${job.deduped ? " deduped" : ""}</small>
    </div>
  `;
}

export function renderMessages() {
  if (!state.messages.length) {
    els.messages.innerHTML = '<div class="empty">Ask about metrics, hooks, comments, creators, or improvements.</div>';
    return;
  }
  els.messages.innerHTML = "";
  for (const message of state.messages) {
    const row = document.createElement("div");
    row.className = `message ${message.role}`;
    row.innerHTML = `<pre>${escapeHtml(message.text || (message.pending ? "Thinking..." : ""))}</pre>`;
    appendCitations(row, message);
    els.messages.append(row);
  }
  els.messages.scrollTop = els.messages.scrollHeight;
}

export function setStatus(status) {
  els.apiStatus.textContent = status;
  els.apiStatus.className = `status status-${status}`;
}

function appendCitations(row, message) {
  const citations = message.result?.citations || [];
  if (citations.length) {
    const chips = document.createElement("div");
    chips.className = "chips";
    chips.innerHTML = citations.slice(0, 8).map((citation) => `<span>${escapeHtml(citation.label)}</span>`).join("");
    row.append(chips);
  }
  if (message.result?.citation_audit) {
    const audit = document.createElement("small");
    audit.className = message.result.citation_audit.valid ? "ok" : "error";
    audit.textContent = `citations ${message.result.citation_audit.valid ? "valid" : "need review"}`;
    row.append(audit);
  }
}
