export function updateLastAssistant(items, token) {
  const copy = [...items];
  const last = copy[copy.length - 1];
  if (last?.role === "assistant") {
    copy[copy.length - 1] = { ...last, text: `${last.text || ""}${token}` };
  }
  return copy;
}

export function finishLastAssistant(items, result) {
  const copy = [...items];
  const last = copy[copy.length - 1];
  if (last?.role === "assistant") {
    copy[copy.length - 1] = { ...last, pending: false, result };
  }
  return copy;
}
