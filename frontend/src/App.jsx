import { useEffect, useState } from "react";

import { api, getJob, loadComparison, streamChat } from "./api.js";
import { ChatPanel } from "./components/ChatPanel.jsx";
import { freshJobForm, IndexPanel } from "./components/IndexPanel.jsx";
import { VideoCard } from "./components/VideoCard.jsx";
import { buildChatPayload, buildExtractIndexPayload, emptyJobForm, prepareJobForm } from "./forms.js";
import { finishLastAssistant, updateLastAssistant } from "./state.js";

export default function App() {
  const [apiStatus, setApiStatus] = useState("checking");
  const [activeComparisonId, setActiveComparisonId] = useState("");
  const [activeComparison, setActiveComparison] = useState(null);
  const [jobForm, setJobForm] = useState(emptyJobForm);
  const [job, setJob] = useState(null);
  const [jobError, setJobError] = useState("");
  const [jobHint, setJobHint] = useState("");
  const [chatQuestion, setChatQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [chatBusy, setChatBusy] = useState(false);
  const [chatError, setChatError] = useState("");

  useEffect(() => {
    checkApi();
  }, []);

  useEffect(() => {
    if (!activeComparisonId) {
      setActiveComparison(null);
      return;
    }
    loadComparison(activeComparisonId).then(setActiveComparison).catch(() => setActiveComparison(null));
  }, [activeComparisonId]);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return undefined;
    const timer = setInterval(async () => {
      try {
        const next = await getJob(job.job_id);
        setJob(next);
        if (next.status === "succeeded") {
          const comparisonId = next.result?.index?.comparison_id || next.result?.comparison_id;
          if (comparisonId) setActiveComparisonId(comparisonId);
        }
      } catch (error) {
        if (String(error.message || "").startsWith("Unknown job:")) {
          setJob(null);
          setJobError("That job no longer exists because the backend restarted. Start a new pipeline run.");
          setJobHint("");
          return;
        }
        setJobHint(`Still running; status polling missed once (${error.message}). Retrying...`);
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [job]);

  async function checkApi() {
    try {
      const health = await api("/health");
      setApiStatus(health.status === "ok" ? "online" : "unknown");
    } catch {
      setApiStatus("offline");
    }
  }

  function handleFresh() {
    setActiveComparisonId("");
    setActiveComparison(null);
    setJob(null);
    setJobError("");
    setJobHint("");
    setChatError("");
    setChatQuestion("");
    setMessages([]);
    setJobForm(freshJobForm());
  }

  async function handleSubmitJob(event) {
    event.preventDefault();
    if (job && ["queued", "running"].includes(job.status)) return;
    setJobError("");
    setJobHint("");
    const prepared = prepareJobForm(jobForm);
    setJobForm(prepared.form);
    setJobHint(prepared.messages.join(" "));
    if (prepared.errors.length) {
      setJobError(prepared.errors.join(" "));
      return;
    }
    try {
      const created = await api("/jobs/extract-index", {
        method: "POST",
        body: JSON.stringify(buildExtractIndexPayload(prepared.form)),
      });
      setJob(created);
    } catch (error) {
      setJobError(error.message);
    }
  }

  async function handleChat(event) {
    event.preventDefault();
    const question = chatQuestion.trim();
    if (!question || chatBusy || !activeComparisonId) return;

    setChatBusy(true);
    setChatError("");
    setChatQuestion("");
    setMessages((items) => [...items, { role: "user", text: question }, { role: "assistant", text: "", pending: true }]);
    try {
      const result = await streamChat(
        buildChatPayload({ activeComparison, activeComparisonId, question }),
        (token) => setMessages((items) => updateLastAssistant(items, token)),
      );
      setMessages((items) => finishLastAssistant(items, result));
    } catch (error) {
      setChatError(error.message);
      setMessages((items) => finishLastAssistant(items, { error: error.message }));
    } finally {
      setChatBusy(false);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <h1>compaRAG</h1>
          <p>Two-video RAG comparison workspace</p>
        </div>
        <div className={`status status-${apiStatus}`}>{apiStatus}</div>
      </header>

      <section className="workspace">
        <aside className="left">
          <IndexPanel
            form={jobForm}
            job={job}
            error={jobError}
            hint={jobHint}
            onChange={setJobForm}
            onFresh={handleFresh}
            onSubmit={handleSubmitJob}
          />
        </aside>

        <section className="center">
          <div className="cards">
            {(activeComparison?.videos || []).map((video) => <VideoCard key={video.video_id} video={video} />)}
            {!activeComparison?.videos?.length && <div className="empty">No comparison loaded.</div>}
          </div>
        </section>

        <ChatPanel
          activeComparisonId={activeComparisonId}
          busy={chatBusy}
          error={chatError}
          messages={messages}
          question={chatQuestion}
          onQuestionChange={setChatQuestion}
          onSubmit={handleChat}
        />
      </section>
    </main>
  );
}
