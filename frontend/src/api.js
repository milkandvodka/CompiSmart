const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8001";

export async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}

export function loadComparison(id) {
  return api(`/comparisons/${encodeURIComponent(id)}`);
}

export function getJob(id) {
  return api(`/jobs/${encodeURIComponent(id)}`);
}

export async function streamChat(payload, onToken) {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(response.statusText);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let doneEvent = null;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      const event = parseSse(block);
      if (!event) continue;
      if (event.type === "token") onToken(event.text || "");
      if (event.type === "error") throw new Error(event.detail || "stream failed");
      if (event.type === "done") doneEvent = event;
    }
  }
  return doneEvent || {};
}

function parseSse(block) {
  const dataLine = block.split("\n").find((line) => line.startsWith("data: "));
  return dataLine ? JSON.parse(dataLine.slice(6)) : null;
}
