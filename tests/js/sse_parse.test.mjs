import assert from "node:assert/strict";
import test from "node:test";

import { createSseParser } from "../../dafx26_demo/static/js/sse_parse.mjs";

function bytes(text) {
  return new TextEncoder().encode(text);
}

test("parser preserves a split SSE frame", () => {
  const parser = createSseParser();
  assert.deepEqual(parser.push(bytes("event: meta\ndata: {\"job")), []);
  assert.deepEqual(parser.push(bytes("_id\":\"x\"}\n\n"))[0], {
    name: "meta",
    data: { job_id: "x" },
  });
});

test("parser accepts CRLF and multiple frames per chunk", () => {
  const parser = createSseParser();
  const events = parser.push(
    bytes("event: midi\r\ndata: {\"kind\":\"note\"}\r\n\r\nevent: status\r\ndata: {\"tokens\":1}\r\n\r\n"),
  );
  assert.equal(events.length, 2);
  assert.equal(events[0].name, "midi");
  assert.equal(events[1].data.tokens, 1);
});

test("parser joins multiple data lines and ignores comments", () => {
  const parser = createSseParser();
  const events = parser.push(
    bytes(": keep-alive\n\nevent: end\ndata: {\"reason\":\ndata: \"stop\"}\n\n"),
  );
  assert.equal(events.length, 1);
  assert.deepEqual(events[0], { name: "end", data: { reason: "stop" } });
});

test("parser reassembles split multibyte UTF-8", () => {
  const parser = createSseParser();
  const encoded = bytes('event: status\ndata: {"msg":"café"}\n\n');
  const splitAt = encoded.indexOf(0xc3);
  assert.ok(splitAt > 0);
  assert.deepEqual(parser.push(encoded.slice(0, splitAt + 1)), []);
  const events = parser.push(encoded.slice(splitAt + 1));
  assert.equal(events[0].data.msg, "café");
});

test("finish rejects malformed JSON and an unterminated frame", () => {
  const bad = createSseParser();
  assert.throws(() => {
    bad.push(bytes("event: end\ndata: {bad}\n\n"));
  }, /JSON/);
  const unfinished = createSseParser();
  unfinished.push(bytes("event: midi\ndata: {\"kind\":\"note\"}"));
  assert.throws(() => unfinished.finish(), /unterminated/);
});
