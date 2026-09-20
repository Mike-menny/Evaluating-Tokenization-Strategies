function parseBlock(block) {
  const lines = block.split(/\r?\n/);
  let name = "message";
  const dataLines = [];
  for (const line of lines) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("event:")) {
      name = line.slice(6).trim();
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).replace(/^ /, ""));
    }
  }
  if (!dataLines.length) {
    return null;
  }
  const payload = dataLines.join("");
  try {
    return { name, data: JSON.parse(payload) };
  } catch (err) {
    throw new Error(`invalid SSE JSON: ${err instanceof Error ? err.message : String(err)}`);
  }
}

export function createSseParser() {
  const decoder = new TextDecoder("utf-8", { fatal: false });
  let buffer = "";

  function takeFrames(text) {
    const events = [];
    const chunks = text.split(/\r?\n\r?\n/);
    const rest = chunks.pop() ?? "";
    for (const block of chunks) {
      const event = parseBlock(block);
      if (event) {
        events.push(event);
      }
    }
    return { events, rest };
  }

  return {
    push(chunk) {
      buffer += decoder.decode(chunk, { stream: true });
      const { events, rest } = takeFrames(buffer);
      buffer = rest;
      return events;
    },
    finish() {
      buffer += decoder.decode();
      if (buffer.trim()) {
        throw new Error("unterminated SSE frame");
      }
    },
  };
}
