import { Message } from "./Message.jsx";

export function ChatPanel({ activeComparisonId, busy, error, messages, question, onQuestionChange, onSubmit }) {
  return (
    <section className="chat panel">
      <div className="panel-head">
        <h2>Chat</h2>
        <span>{activeComparisonId || "no comparison"}</span>
      </div>
      <div className="messages">
        {messages.map((message, index) => (
          <Message key={index} message={message} />
        ))}
        {!messages.length && <div className="empty">Ask about metrics, hooks, comments, creators, or improvements.</div>}
      </div>
      <form className="chat-form" onSubmit={onSubmit}>
        <textarea
          value={question}
          onChange={(event) => onQuestionChange(event.target.value)}
          placeholder="Ask a hard question..."
        />
        <button disabled={busy || !activeComparisonId}>{busy ? "Answering..." : "Send"}</button>
      </form>
      {error && <p className="error">{error}</p>}
    </section>
  );
}
