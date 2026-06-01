export function Message({ message }) {
  const meta = message.result;
  return (
    <div className={`message ${message.role}`}>
      <pre>{message.text || (message.pending ? "Thinking..." : "")}</pre>
      {meta?.citations?.length > 0 && (
        <div className="chips">
          {meta.citations.slice(0, 8).map((citation) => (
            <span key={citation.label}>{citation.label}</span>
          ))}
        </div>
      )}
      {meta?.citation_audit && (
        <small className={meta.citation_audit.valid ? "ok" : "error"}>
          citations {meta.citation_audit.valid ? "valid" : "need review"}
        </small>
      )}
    </div>
  );
}
