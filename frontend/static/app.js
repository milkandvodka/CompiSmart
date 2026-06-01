import { api, streamChat } from "./js/api.js";
import { els, state } from "./js/state.js";
import { renderJob, renderMessages, renderVideos, setStatus } from "./js/render.js";
import { newComparisonId, stableKey, value } from "./js/utils.js";
import { DEFAULT_COMMENT_LIMIT, EMBEDDING_HINTS, prepareJobForm } from "./js/validation.js";

els.jobForm.addEventListener("submit", submitJob);
els.freshComparison.addEventListener("click", resetWorkspace);
els.chatForm.addEventListener("submit", submitChat);
document.querySelector("#embeddingModel").addEventListener("change", updateEmbeddingHint);

checkApi();
resetWorkspace();
updateEmbeddingHint();

async function checkApi() {
  try {
    const health = await api("/health");
    setStatus(health.status === "ok" ? "online" : "unknown");
  } catch {
    setStatus("offline");
  }
}

async function setActiveComparison(id) {
  state.activeComparisonId = id;
  els.chatComparison.textContent = id || "no comparison";
  els.sendChat.disabled = !id;
  if (!id) {
    state.activeComparison = null;
    renderVideos();
    return;
  }
  state.activeComparison = await api(`/comparisons/${encodeURIComponent(id)}`);
  renderVideos();
}

async function submitJob(event) {
  event.preventDefault();
  if (state.job && ["queued", "running"].includes(state.job.status)) return;
  els.jobError.textContent = "";
  els.jobHint.textContent = "";
  const prepared = prepareJobForm(readJobForm());
  writeJobForm(prepared.form);
  els.jobHint.textContent = prepared.messages.join(" ");
  if (prepared.errors.length) {
    els.jobError.textContent = prepared.errors.join(" ");
    return;
  }
  try {
    state.job = await api("/jobs/extract-index", {
      method: "POST",
      body: JSON.stringify(buildJobPayload(prepared.form)),
    });
    renderJob();
    pollJob();
  } catch (error) {
    els.jobError.textContent = error.message;
  }
}

function readJobForm() {
  return {
    comparisonId: value("#comparisonId"),
    youtubeUrl: value("#youtubeUrl"),
    instagramUrl: value("#instagramUrl"),
    embeddingModel: value("#embeddingModel"),
    maxComments: DEFAULT_COMMENT_LIMIT,
    requireTranscripts: true,
    commentIntelligence: "evidence",
    creativeFeatures: "evidence",
  };
}

function writeJobForm(form) {
  document.querySelector("#comparisonId").value = form.comparisonId;
  document.querySelector("#youtubeUrl").value = form.youtubeUrl;
  document.querySelector("#instagramUrl").value = form.instagramUrl;
  document.querySelector("#embeddingModel").value = form.embeddingModel;
  updateEmbeddingHint();
}

function buildJobPayload(form) {
  return {
    comparison_id: form.comparisonId,
    youtube_url: form.youtubeUrl,
    instagram_url: form.instagramUrl,
    extraction: {
      fetch_comments: true,
      max_comments: Number(form.maxComments) || 100,
      comment_time_budget_seconds: 60,
      instagrapi_settings: ".cache/instagrapi-session.json",
      asr_provider: "auto",
      asr_model: "base",
      asr_timeout_seconds: 90,
      require_transcripts: form.requireTranscripts,
    },
    index: {
      embedding_model: form.embeddingModel,
      allow_embedding_download: true,
      comment_intelligence: form.commentIntelligence,
      creative_features: form.creativeFeatures,
    },
    idempotency_key: stableKey({
      id: form.comparisonId,
      yt: form.youtubeUrl,
      ig: form.instagramUrl,
      comments: form.maxComments,
      embedding: form.embeddingModel,
    }),
  };
}

function updateEmbeddingHint() {
  const model = value("#embeddingModel");
  els.embeddingHint.textContent = EMBEDDING_HINTS[model] || "";
}

function pollJob() {
  clearInterval(state.jobTimer);
  state.jobTimer = setInterval(async () => {
    if (!state.job) return;
    try {
      state.job = await api(`/jobs/${encodeURIComponent(state.job.job_id)}`);
      renderJob();
      if (["succeeded", "failed"].includes(state.job.status)) {
        clearInterval(state.jobTimer);
        const comparisonId = state.job.result?.index?.comparison_id || state.job.result?.comparison_id;
        if (comparisonId) await setActiveComparison(comparisonId);
      }
    } catch (error) {
      els.jobError.textContent = error.message;
    }
  }, 1500);
}

async function submitChat(event) {
  event.preventDefault();
  const question = els.chatQuestion.value.trim();
  if (!state.activeComparisonId) {
    els.chatError.textContent = "Index a new comparison or select an existing one before chatting.";
    return;
  }
  if (!question || state.chatBusy) return;

  state.chatBusy = true;
  els.sendChat.disabled = true;
  els.sendChat.textContent = "Answering...";
  els.chatQuestion.value = "";
  els.chatError.textContent = "";
  state.messages.push({ role: "user", text: question }, { role: "assistant", text: "", pending: true });
  renderMessages();

  try {
    const result = await streamChat(buildChatPayload(question), (token) => {
      const last = state.messages[state.messages.length - 1];
      last.text += token;
      renderMessages();
    });
    Object.assign(state.messages[state.messages.length - 1], { pending: false, result });
  } catch (error) {
    els.chatError.textContent = error.message;
    Object.assign(state.messages[state.messages.length - 1], { pending: false, text: `Error: ${error.message}` });
  } finally {
    state.chatBusy = false;
    els.sendChat.disabled = false;
    els.sendChat.textContent = "Send";
    renderMessages();
  }
}

function resetWorkspace() {
  clearInterval(state.jobTimer);
  state.activeComparisonId = "";
  state.activeComparison = null;
  state.job = null;
  state.messages = [];
  state.chatBusy = false;
  els.jobError.textContent = "";
  els.jobHint.textContent = "";
  els.chatError.textContent = "";
  els.chatQuestion.value = "";
  writeJobForm({
    comparisonId: newComparisonId(),
    youtubeUrl: "",
    instagramUrl: "",
    embeddingModel: "quality",
    maxComments: DEFAULT_COMMENT_LIMIT,
    requireTranscripts: true,
    commentIntelligence: "evidence",
    creativeFeatures: "evidence",
  });
  setActiveComparison("");
  renderJob();
  renderMessages();
}

function buildChatPayload(question) {
  return {
    comparison_id: state.activeComparisonId,
    question,
    thread_id: `ui_${state.activeComparisonId}`,
    idempotency_key: stableKey({ id: state.activeComparisonId, question }),
    options: {
      llm: "auto",
      embedding_model: state.activeComparison?.embedding_model || "quality",
      retrieval_mode: "hybrid",
      memory_backend: "local",
      allow_embedding_download: true,
    },
  };
}
