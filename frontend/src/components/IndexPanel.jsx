import { clampPercent, newComparisonId } from "../format.js";
import { embeddingHints } from "../forms.js";

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
      <small>
        {job.job_id}
        {job.deduped ? " deduped" : ""}
      </small>
    </div>
  );
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
    maxComments: 100,
    requireTranscripts: true,
    commentIntelligence: "evidence",
    creativeFeatures: "evidence",
  };
}
