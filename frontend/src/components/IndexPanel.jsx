import { clampPercent, newComparisonId } from "../format.js";
import { asrModelHints, embeddingHints } from "../forms.js";

export function IndexPanel({ form, job, error, hint, onChange, onFresh, onSubmit }) {
  const isWorking = job && ["queued", "running"].includes(job.status);
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Index Videos</h2>
        <button type="button" className="secondary" onClick={onFresh}>
          Fresh
        </button>
      </div>
      <form className="stack" onSubmit={onSubmit} autoComplete="off">
        <label>
          Comparison ID
          <input value={form.comparisonId} onChange={(event) => update(onChange, form, "comparisonId", event.target.value)} />
        </label>
        <label>
          YouTube Shorts URL
          <input
            value={form.youtubeUrl}
            onChange={(event) => update(onChange, form, "youtubeUrl", event.target.value)}
            placeholder="https://youtube.com/shorts/..."
          />
        </label>
        <label>
          Instagram URL
          <input
            value={form.instagramUrl}
            onChange={(event) => update(onChange, form, "instagramUrl", event.target.value)}
            placeholder="https://www.instagram.com/p/..."
          />
        </label>
        <div className="two-col">
          <label>
            Embeddings
            <select value={form.embeddingModel} onChange={(event) => update(onChange, form, "embeddingModel", event.target.value)}>
              <option value="quality">BGE-M3 - slow, most accurate</option>
              <option value="balanced">E5 base - balanced</option>
              <option value="fast">MiniLM - fastest, lower recall</option>
            </select>
            <small className="hint">{embeddingHints[form.embeddingModel]}</small>
          </label>
          <label>
            Whisper Model
            <select value={form.asrModel} onChange={(event) => update(onChange, form, "asrModel", event.target.value)}>
              <option value="base">Base - fastest, rougher</option>
              <option value="small">Small - better, slower</option>
              <option value="medium">Medium - best local, slow</option>
            </select>
            <small className="hint">{asrModelHints[form.asrModel]}</small>
          </label>
        </div>
        <button disabled={isWorking}>{isWorking ? "Working..." : "Run Pipeline"}</button>
      </form>
      {hint && <p className="hint">{hint}</p>}
      {error && <p className="error">{error}</p>}
      {job && <JobProgress job={job} />}
    </section>
  );
}

function JobProgress({ job }) {
  const progressPercent = clampPercent(job?.progress?.percent);
  const events = [...(job?.progress_events || [])].slice(-8).reverse();
  return (
    <div className="job">
      <div className="job-row">
        <strong>{job.status}</strong>
        <span>{job.progress?.stage || "queued"}</span>
      </div>
      <div className="bar">
        <span style={{ width: `${progressPercent}%` }} />
      </div>
      <p>{job.progress?.message || "Waiting for progress..."}</p>
      <JobDetails details={job.progress?.details} />
      {events.length > 1 && (
        <ol className="job-events">
          {events.map((event, index) => (
            <li key={`${event.updated_at || index}-${event.stage || index}`}>
              <span>{formatStage(event.stage)}</span>
              <small>{event.message}</small>
            </li>
          ))}
        </ol>
      )}
      <small>
        {job.job_id}
        {job.deduped ? " deduped" : ""}
      </small>
    </div>
  );
}

function JobDetails({ details }) {
  const items = Object.entries(details || {}).filter(([, value]) => value !== null && value !== undefined && value !== "");
  if (!items.length) return null;
  return (
    <dl className="job-details">
      {items.slice(0, 8).map(([key, value]) => (
        <div key={key}>
          <dt>{formatStage(key)}</dt>
          <dd>{String(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function formatStage(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function update(onChange, form, field, value) {
  onChange({ ...form, [field]: value });
}

export function freshJobForm() {
  return {
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
}
