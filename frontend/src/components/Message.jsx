export function Message({ message }) {
  const meta = message.result;
  return (
    <div className={`message ${message.role}`}>
      <pre>{message.text || (message.pending ? "Thinking..." : "")}</pre>
      {meta?.citations?.length > 0 && (
        <details className="source-details">
          <summary>Sources ({meta.citations.length})</summary>
          <div className="chips">
            {meta.citations.slice(0, 8).map((citation) => (
              <span key={citation.label}>{citation.label}</span>
            ))}
          </div>
        </details>
      )}
      {meta?.citation_audit && !meta.citation_audit.valid && (
        <small className="error">citations need review</small>
      )}
    </div>
  );
}
